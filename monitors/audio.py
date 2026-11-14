"""NPU Whisper voice command producer with rolling audio window for microphone-3."""
import asyncio
from collections import deque
import json
import logging
from pathlib import Path
import re
import time
import traceback

import numpy as np

from core.runtime_events import EventKind, ProducerContext

LOG = logging.getLogger(__name__)


def normalize_transcript(text: str) -> str:
    cleaned = re.sub(r"[^a-z0-9\s]", " ", text.lower())
    return re.sub(r"\s+", " ", cleaned).strip()


def parse_voice_command(raw_text: str) -> dict | None:
    """Map raw Whisper transcript directly to hello / start / stop / dance / new_policy."""
    norm = normalize_transcript(raw_text)
    if not norm:
        return None

    # 1. "New policy <rule>" (or "New policies <rule>")
    policy_match = re.search(
        r"\bnew\s+polic(?:y|ies)[:,.\-\s]+(.+)$",
        raw_text.strip(),
        flags=re.IGNORECASE,
    )
    if policy_match:
        policy_text = policy_match.group(1).strip()
        if policy_text:
            return {
                "command": "new_policy",
                "policy": policy_text,
                "raw_text": raw_text.strip(),
            }

    # 2. "Start" / "Starts"
    if re.search(r"\b(?:starts?|started|starting)\b", norm):
        return {"command": "start", "raw_text": raw_text.strip()}

    # 3. "Stop" / "Stops"
    if re.search(r"\b(?:stops?|stopped|stopping)\b", norm):
        return {"command": "stop", "raw_text": raw_text.strip()}

    # 4. "Dance" / "Dancer" / "Dancing"
    if re.search(r"\b(?:dances?|dancer|dancing)\b", norm):
        return {"command": "dance", "raw_text": raw_text.strip()}

    # 5. "Hello" / "Halo" / "Hi" / "Hey"
    if re.search(r"\b(?:hello|halo|helo|hi|hey)\b", norm):
        return {"command": "hello", "raw_text": raw_text.strip()}

    return None


def _prepare_16k_mono(raw_bytes: bytes, sample_rate: int, num_channels: int) -> tuple[np.ndarray, float]:
    """Convert raw PCM16 bytes to 16 kHz mono float32 in [-1, 1] and compute RMS amplitude."""
    if not raw_bytes:
        return np.empty(0, dtype=np.float32), 0.0
    pcm = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32)
    if num_channels > 1:
        pcm2d = pcm.reshape(-1, num_channels)
        best_ch = int(np.argmax(np.max(np.abs(pcm2d), axis=0)))
        pcm = pcm2d[:, best_ch]

    rms = float(np.sqrt(np.mean(pcm ** 2))) if len(pcm) else 0.0
    if sample_rate != 16000 and len(pcm):
        old_x = np.linspace(0, len(pcm) - 1, len(pcm))
        new_x = np.linspace(0, len(pcm) - 1, int(round(len(pcm) * 16000 / sample_rate)))
        pcm = np.interp(new_x, old_x, pcm)

    return np.clip(pcm / 32768.0, -1.0, 1.0).astype(np.float32), rms


def _resolve_model_dir(settings: dict, root: Path) -> Path:
    if "model_path" in settings:
        return (root / settings["model_path"]).resolve()
    base_dir = (root / "models/whisper-base-ov").resolve()
    if base_dir.exists():
        return base_dir
    return (root / "models/whisper-tiny-ov").resolve()


def _load_whisper_pipeline(settings: dict, root: Path):
    import openvino as ov
    import openvino_genai as genai

    device = str(settings.get("device", "NPU")).upper()
    if device not in ("NPU", "CPU"):
        raise ValueError("audio.device must be explicit NPU or CPU")
    core = ov.Core()
    if device not in core.available_devices:
        raise RuntimeError(f"Requested speech device {device} is unavailable")

    model_dir = _resolve_model_dir(settings, root)
    cache_tag = model_dir.name
    cache_dir = (root / settings.get("cache_dir", f"/tmp/speech-cache/{cache_tag}-npu")).resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)

    pipe = genai.WhisperPipeline(str(model_dir), device, CACHE_DIR=str(cache_dir))
    gen_cfg = pipe.get_generation_config()
    gen_cfg.max_new_tokens = int(settings.get("max_new_tokens", 32))
    if hasattr(gen_cfg, "language"):
        gen_cfg.language = str(settings.get("language", "<|en|>"))
    if hasattr(gen_cfg, "task"):
        gen_cfg.task = "transcribe"
    return pipe, gen_cfg, model_dir.name


async def _listen_loop(context: ProducerContext, root: Path):
    from viam.components.audio_in import AudioIn
    from robot.viam_hw import connect, disconnect, load_credentials

    settings = context.settings
    mic_name = str(settings.get("mic_name", "microphone-3"))
    step_s = float(settings.get("step_seconds", 1.0))
    window_s = float(settings.get("window_seconds", 3.5))
    rms_threshold = float(settings.get("rms_threshold", 400.0))
    cooldown_s = float(settings.get("cooldown_s", 1.5))
    debug_print = bool(settings.get("print_transcripts", True))

    if debug_print:
        print(f"[AUDIO] Loading WhisperPipeline on {settings.get('device', 'NPU')}...", flush=True)
    pipe, gen_cfg, model_name = await asyncio.to_thread(_load_whisper_pipeline, settings, root)

    machine = await connect(load_credentials(root / ".env"))
    last_command_key = None
    last_command_ts = 0.0
    max_samples = int(16000 * window_s)
    ring = deque(maxlen=max_samples)

    try:
        mic = AudioIn.from_robot(machine, mic_name)
        props = await mic.get_properties()
        sr = getattr(props, "sample_rate_hz", 16000) or 16000
        ch = getattr(props, "num_channels", 1) or 1
        if debug_print:
            print(
                f"[AUDIO] Ready ({model_name}, mic={mic_name}, rate={sr}Hz, ch={ch}, "
                f"window={window_s}s/{step_s}s, rms_gate={rms_threshold})",
                flush=True,
            )

        async for _ in await mic.get_audio(codec="pcm16", duration_seconds=0.2, previous_timestamp_ns=0):
            if context.stop.is_set():
                return

        while not context.stop.is_set():
            chunks = []
            stream = await mic.get_audio(codec="pcm16", duration_seconds=step_s, previous_timestamp_ns=0)
            async for chunk in stream:
                if context.stop.is_set():
                    break
                chunks.append(chunk.audio.audio_data)

            if context.stop.is_set():
                break

            raw_bytes = b"".join(chunks)
            step_f32, step_rms = _prepare_16k_mono(raw_bytes, sr, ch)
            if len(step_f32) == 0:
                continue

            ring.extend(step_f32.tolist())
            if step_rms < rms_threshold or len(ring) < int(16000 * 1.5):
                continue

            window_audio = list(ring)
            result = await asyncio.to_thread(pipe.generate, window_audio, gen_cfg)
            transcript = " ".join(result.texts).strip()
            parsed = parse_voice_command(transcript) if transcript else None

            if debug_print:
                status = f"MATCH -> {parsed['command']}" if parsed else "ignored"
                print(f"[AUDIO] rms={step_rms:.1f} | transcript={transcript!r} ({status})", flush=True)

            if parsed is None:
                continue

            now_mono = time.monotonic()
            dedup_key = (parsed["command"], parsed.get("policy", ""))
            if dedup_key == last_command_key and (now_mono - last_command_ts) < cooldown_s:
                continue
            last_command_key = dedup_key
            last_command_ts = now_mono

            ring.clear()
            context.publish(EventKind.NOTICE, json.dumps(parsed))
    finally:
        await disconnect(machine)


def run(context: ProducerContext):
    """Entrypoint invoked by EventRuntime in a background thread."""
    if context.fake:
        context.stop.wait()
        return

    root = Path(__file__).resolve().parent.parent
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(_listen_loop(context, root))
    except Exception:
        print("[AUDIO ERROR] Listener crashed:\n" + traceback.format_exc(), flush=True)
        raise
    finally:
        pending = asyncio.all_tasks(loop)
        for task in pending:
            task.cancel()
        if pending:
            loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
        loop.close()
