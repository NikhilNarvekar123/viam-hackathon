# PREP_TASKS.md: pre-event build list

Work through these tasks **one at a time, in order**. For each task:

1. Read AGENTS.md if you haven't already this session.
2. Show me a short plan and the files you'll touch, and wait for my OK.
3. Build it.
4. Run `python main.py --fake --n 10` and `pytest`, and show me the real output.
5. List the files you changed, then stop so I can review and commit.
6. Tick the task's checkbox here when I confirm.

## Rules for tonight

- **Heavy packages are NOT installed tonight:** `anomalib`, `openvino`, `openvino-genai`, `viam-sdk`, and torch. Write the code that uses them anyway, but **import them lazily**, inside the class or function that needs them. That way `main.py --fake` and all tests still run without them.
- Code that depends on those packages can't be run tonight. Keep it small and readable. Wherever you're unsure of an exact API call or signature, add a `# VERIFY:` comment saying what to check, so we can find them all tomorrow with `grep -rn "VERIFY:"`.
- Mark every value that comes from the event (resource names, poses, defect types, actions, thresholds) with `# TODO(event):`.
- Every pure-logic piece (math, parsing, thresholds, prompt building, image overlays, config handling) gets pytest tests that run tonight.
- Each real implementation is selectable in `config/config.yaml`, alongside the existing stub or fake. Fakes and stubs stay the default.
- Small dependencies like `python-dotenv` are fine to add and install. Ask before adding anything else.

---

## [x] 1. Viam hardware layer (written, not run)

There's no Viam account or machine tonight, so write this against the SDK without testing it live.

- `robot/viam_hw.py`: connect to the machine using `VIAM_ADDRESS`, `VIAM_API_KEY`, and `VIAM_API_KEY_ID` loaded from `.env`. Add `ViamCamera`, `ViamArm`, and `ViamGripper` wrappers that implement the **same interfaces** as the fakes in `core/types.py` / `robot/fakes.py`.
  - Camera: color image as a PIL image, plus depth as a numpy array if a depth camera name is set in config.
  - Arm: `move_to(pose_name)` using joint positions from `config/poses.json` (`move_to_joint_positions`), and `move_to_xyz(x, y, z)` with the gripper pointing down (via `move_to_position` or the motion service, `MotionClient.from_robot(machine, "builtin")`). Also `get_joint_positions()`, `get_end_position()`, and `stop()`.
  - Gripper: `open()` and `grab()`.
- Add a `.env.example` with the three variable names (no values). Make sure `.env` is in `.gitignore`.
- `main.py` real mode (no `--fake`): build the Viam wrappers from config instead of exiting. **Refuse to move the arm unless `--confirm-motion` is passed.** Without it, real mode may connect, capture, detect, and decide, but must skip all arm and gripper calls and log what it would have done.
- On Ctrl+C or any exception during motion: call `arm.stop()`, disconnect cleanly, and exit.

**Done when:** fake mode still passes, real-mode code paths are complete with `VERIFY:` markers, and `python main.py` without credentials fails with a clear "missing .env values" message rather than a stack trace.

## [x] 2. Pick step and sort sequence

- `robot/pick.py`: `get_pick_target(frame, config)` returns either a named pose (config `pick_mode: fixed`) or an x/y/z target (`pick_mode: vision`, using the calibration from task 8; stub it until that exists).
- Sort sequence: home → above_pick → pick → grab → above_pick → above_<bin> → open → home. In vision mode, "above_pick" is the target plus a configurable z offset.
- Bins come from `action_to_bin` in config. Items that pass detection always go to the box.
- Works with both the fake arm and the Viam arm through the shared interface.

**Done when:** fake runs log the full sequence for both bins and both pick modes, and the sequence logic has tests.

## [x] 3. Pose teaching tool

- `robot/teach.py`:
  - `python robot/teach.py <name>` reads the arm's current joint positions (and end position, for reference) and saves them under `<name>` in `config/poses.json`. Keep the other poses, and back up the previous file first.
  - `--list` prints all saved poses.
  - `--goto <name>` moves to a pose, **only with `--confirm-motion`**.
  - `--fake` uses the fake arm.
- Expected pose names: `home`, `above_pick`, `pick`, `above_box`, `above_reject` (plus any bins added in config).

**Done when:** `--fake` save, list, and goto all work, and saving merges into existing poses instead of overwriting them.

## [x] 4. Data capture tool

Tomorrow we need training photos fast.

- `tools/capture.py --out data/<item>/good --n 50`: grabs frames from the camera (fake or Viam), crops to the ROI from config, and saves numbered images. Pressing Enter takes the next photo, so someone can move or swap the item between shots. A `--interval` option captures every N seconds instead.
- `--preview` saves one uncropped and one cropped image so we can check the ROI.

**Done when:** it works end to end with the fake camera.

## [x] 5. Anomalib: train, threshold, export, detector

Use the **Anomalib 2.x API** (`anomalib.data.Folder`, `anomalib.models.Patchcore` / `Padim` / `EfficientAd`, `anomalib.engine.Engine`, export to OpenVINO). Ignore 0.x/1.x tutorials that use `config.yaml` plus `tools/` scripts. CPU torch only.

- `anomaly/train.py --data data/<item> --model patchcore --image-size 256`:
  - Train on `good/`, with `defect/` optional.
  - Export OpenVINO IR to `models/anomaly/<item>_<model>/`.
  - Score every available image and print the score stats for good vs. defect.
  - Suggest a threshold, defaulting to the highest good score plus a configurable margin, or the best separating value if defect images exist.
  - Print the false accepts and false rejects at that threshold.
  - Save `threshold` and the model path to a small JSON file next to the model, and print the lines to paste into config. Don't edit `config.yaml` automatically.
- `anomaly/threshold.py`: the threshold logic as pure functions, with tests.
- `anomaly/detector.py`: an OpenVINO-based detector implementing the Detector interface.
  - Device from config, falling back to CPU with a warning if it's not available.
  - Set `CACHE_DIR`. Reshape to a static input shape when the device is NPU.
  - Warm up once at startup.
  - Crop to the ROI, then apply exactly the same preprocessing as training. Add `VERIFY:` on whether the exported model already includes normalization.
  - Return score, heatmap, and `is_anomalous` against the config threshold.
- Config: `detector: stub | openvino`.

**Done when:** the code is complete, threshold tests pass, and fake mode is unchanged.

## [x] 6. VLM decider

- `vlm/overlay.py`: given a crop and a heatmap, produce the image the VLM sees: a side-by-side of the original crop and a red-tinted heatmap overlay. Pure numpy/PIL, with tests.
- `vlm/prompt.py`: build the prompt from the template in config, filling in `{defect_types}` and `{allowed_actions}`. The template must demand JSON only, give one example reply, and keep the `reason` short. Tests included.
- `vlm/decider.py`: an OpenVINO GenAI decider implementing the Decider interface.
  - `openvino_genai.VLMPipeline(model_path, device)`, device from config with CPU fallback.
  - Generate with greedy decoding and `max_new_tokens` from config.
  - Warm up once at startup.
  - Every reply goes through `vlm/parse.py`. Record the latency.
- `vlm/rules.py`: a no-model fallback decider that maps every flagged item to `reject`. Use it if the VLM won't load, so sorting still works.
- `vlm/try_vlm.py --image <path>`: run the configured decider on one image and print the prompt, raw output, parsed decision, and latency.
- Config: `decider: stub | rules | openvino`.

**Done when:** overlay, prompt, and parse tests pass; the rules decider works in fake mode; and the OpenVINO decider is complete with `VERIFY:` markers.

## [ ] 7. Device benchmark

- `bench/devices.py`: for each model (the anomaly IR and the VLM) on each device in `ov.Core().available_devices`, measure:
  - compile/load time
  - warm-up time
  - mean, p50, and p95 latency over N runs (the VLM uses one fixed image and prompt; also report tokens/second if available)
- If a model fails to compile on a device, record `failed: <short error>` and continue.
- Print a markdown table and save JSON to `logs/bench_<timestamp>.json`.
- `--pipeline` mode: run the full `main.py` loop on sample images with a given device assignment, for example `--anomaly-device NPU --vlm-device GPU`, and report per-item latency, so we can compare combinations.

**Done when:** the code is complete, and it exits cleanly with a clear message when openvino isn't installed.

## [ ] 8. Calibration for scattered items (optional, only if time allows)

- `robot/locate.py`: find items in an image by comparing against a saved empty-workspace photo (and/or a depth threshold). Return each item's pixel center, bounding box, and angle. Tests use synthetic images.
- `robot/calibrate.py`: collect pairs of marker pixel (u, v) and arm (x, y), where the pixel is entered by hand or auto-detected from a bright marker and the arm position is read from the arm. Fit a pixel-to-arm mapping with numpy (affine least squares, with homography as an option), print the reprojection error, and save it to `config/calibration.json`. Provide `pixel_to_arm(u, v)`.
- Tests with synthetic point sets, including a check that the error is near zero for an exact affine transform.
- The vision pick mode from task 2 uses this. Fixed pick mode stays the default.

## [ ] 9. Run report for judging

- `tools/report.py logs/run_*.jsonl`:
  - accuracy, confusion matrix, false accepts (defects into the box), false rejects, and invalid VLM outputs
  - response accuracy, if expected responses were logged
  - mean and p95 latency per stage, plus items per minute
- Output markdown we can paste into DESIGN.md.
- Real runs need ground truth, so add `--ask-ground-truth` to `main.py`. It prompts the operator after each item for good/defect and, optionally, the expected response, and logs the answers.

## [ ] 10. Runbook and design-doc skeleton

- `RUNBOOK.md`: a day-of checklist with exact commands, in order:
  1. Setup: venv, install, and the device check (`python3 -c "import openvino as ov; print(ov.Core().available_devices)"`).
  2. Fill in `.env` and the resource names in config.
  3. Teach poses.
  4. Set the ROI preview, then capture training data.
  5. Train and set the threshold.
  6. Try the VLM.
  7. Run fake, then real without motion, then real with `--confirm-motion`.
  8. Run the benchmark, then the report.
  - Include a "`grep -rn VERIFY:` and fix these first" step.
  - Add an "if X breaks" section: NPU fails → GPU/CPU; VLM too slow → GPU, fewer tokens, or a smaller model; VLM won't load → `decider: rules`; arm misses picks → re-teach poses, open the gripper wider.
- `DESIGN.md`: a skeleton for the short design explanation.
  - A Mermaid pipeline diagram (capture → detect → decide → sort).
  - A device-split table with rationale and blanks for the benchmark numbers.
  - Blanks for accuracy and latency results.
  - A short section on failure handling (JSON fallback, rules decider, CPU fallback, motion safety).

## [ ] 11. Stretch: overlap stages

- While the arm sorts item N, capture and detect item N+1 (asyncio tasks). Keep the default sequential, with overlap behind a config flag.
- Log per-item latency with and without overlap, so the difference can go in DESIGN.md.