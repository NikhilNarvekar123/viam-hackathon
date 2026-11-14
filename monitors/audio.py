"""NPU Whisper voice command producer with rolling audio window for microphone-3."""
import asyncio
from collections import deque
import json
import logging
import math
from pathlib import Path
import time
import traceback

import numpy as np

from core.runtime_events import EventKind, ProducerContext
from core.voice_control import normalize_transcript, parse_voice_command

LOG = logging.getLogger(__name__)


def _prepare_16k_mono(raw_bytes: bytes, sample_rate: int, num_channels: int) -> tuple[np.ndarray, float]:
    """Convert raw PCM16 bytes to 16 kHz mono float32 in [-1, 1] and compute RMS amplitude."""
    if sample_rate <= 0 or num_channels < 1:
        raise ValueError('Invalid microphone sample rate/channels')
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
    for candidate in ("models/whisper-small-ov", "models/whisper-base-ov", "models/whisper-tiny-ov"):
        cand_dir = (root / candidate).resolve()
        if cand_dir.exists():
            return cand_dir
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
    cache_dir = (root / settings.get("cache_dir", f"/tmp/speech-cache/{cache_tag}-{device.lower()}")).resolve()
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
    timeout_s = float(settings.get('io_timeout_s', 10.0))
    if (not all(math.isfinite(v) for v in (step_s, window_s, rms_threshold, cooldown_s, timeout_s))
            or not 0 < step_s <= window_s <= 30 or window_s < 1.5
            or not 0 <= rms_threshold <= 32768 or not 0 <= cooldown_s <= 30
            or timeout_s <= max(step_s, 0.2)):
        raise ValueError('Invalid audio capture/window/timeout settings')

    if debug_print:
        print(f"[AUDIO] Loading WhisperPipeline on {settings.get('device', 'NPU')}...", flush=True)
    pipe, gen_cfg, model_name = await asyncio.to_thread(_load_whisper_pipeline, settings, root)

    machine = await asyncio.wait_for(connect(load_credentials(root / ".env")), timeout_s)
    last_command_key = None
    last_command_ts = 0.0
    sequence = 0
    commands_enabled = bool(settings.get('commands_enabled', False))
    max_samples = int(16000 * window_s)
    ring = deque(maxlen=max_samples)
    sample_times = deque(maxlen=max_samples)

    try:
        mic = AudioIn.from_robot(machine, mic_name)
        props = await asyncio.wait_for(mic.get_properties(), timeout_s)
        sr = getattr(props, "sample_rate_hz", 16000) or 16000
        ch = getattr(props, "num_channels", 1) or 1
        if debug_print:
            print(
                f"[AUDIO] Ready ({model_name}, mic={mic_name}, rate={sr}Hz, ch={ch}, "
                f"window={window_s}s/{step_s}s, rms_gate={rms_threshold})",
                flush=True,
            )

        async def capture(seconds):
            chunks = []
            stream = await mic.get_audio(codec="pcm16", duration_seconds=seconds, previous_timestamp_ns=0)
            async for chunk in stream:
                if context.stop.is_set():
                    break
                chunks.append(chunk.audio.audio_data)
            return b"".join(chunks)

        await asyncio.wait_for(capture(0.2), timeout_s)
        if commands_enabled and not context.publish(EventKind.AUDIO_READY, json.dumps(
                dict(model=model_name, device=settings.get('device', 'NPU'), microphone=mic_name))):
            return
        while not context.stop.is_set():
            capture_started = time.monotonic()
            raw_bytes = await asyncio.wait_for(capture(step_s), timeout_s)
            if context.stop.is_set():
                break
            step_f32, step_rms = _prepare_16k_mono(raw_bytes, sr, ch)
            if len(step_f32) == 0:
                continue

            ring.extend(step_f32.tolist())
            sample_times.extend([capture_started] * len(step_f32))
            if step_rms < rms_threshold or len(ring) < int(16000 * 1.5):
                continue

            # Use the oldest acquisition in this window, not inference completion:
            # buffered pre-stop speech must never count as a fresh confirming start.
            heard_at = sample_times[0]
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
            sample_times.clear()
            sequence += 1
            parsed.update(sequence=sequence, heard_at=heard_at)
            kind = EventKind.VOICE_COMMAND if commands_enabled else EventKind.NOTICE
            if not context.publish(kind, json.dumps(parsed)):
                return
    finally:
        await asyncio.wait_for(disconnect(machine), timeout_s)


async def _run_until_stopped(context, root):
    """Cancel blocked microphone waits on shutdown; native inference cannot be forcibly killed."""
    task = asyncio.create_task(_listen_loop(context, root))
    try:
        while not task.done() and not context.stop.is_set():
            await asyncio.sleep(0.05)
        if context.stop.is_set():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        else:
            await task  # Propagate producer faults to EventRuntime.
    finally:
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)


def run(context: ProducerContext):
    """Entrypoint invoked by EventRuntime in a background thread."""
    if context.fake:
        context.stop.wait()
        return

    root = Path(__file__).resolve().parent.parent
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(_run_until_stopped(context, root))
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
