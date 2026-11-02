# Running Qwen3-Omni: from zero to an A8 pilot

Status 2026-11-02: everything below up to step 3 is tested on the real classes with tiny configs on CPU. **No
real weights have run and no GPU has been used.** Steps 3 onward are written from the code and have not
been run. "Unverified" is listed at the end.

## 0. Memory comes first

The thinker is **63.4 GB at bf16** (31.7B parameters, 29B of them MoE experts, which a forward pass reads only in
part but which must all be resident). That does not fit a 48 GB RTX A6000.

| GPU | Plan |
|---|---|
| ≥ ~72 GB (A100 80 GB, H100 80 GB [VERIFY the card]) | One rank per GPU, as specified |
| 48 GB (rtxa6000, l40s) | One rank spans 2 GPUs: `--gpus-per-rank 2`, `device_map="balanced"`. 4 GPUs give 2 ranks |
| 24 GB (rtxa5000) | Not supported |

The ~72 GB figure is an estimate, weights plus an allowance for the KV cache, towers and scoring. The smoke run measures it.

## 1. Environment

```bash
cd AVRSI
PY=/fs/nexus-projects/brain_project/miniconda3/envs/gama2/bin/python   # or a fresh env from requirements-qwen3.txt
TMPDIR=<a dir with space> $PY -m pip install --no-cache-dir --no-deps --target .venv/run_lib \
    peft==0.21.2 av==17.1.0 imageio-ffmpeg==0.8.0
export PYTHONPATH=$PWD/.venv/run_lib:$PWD
PYTHONPATH=.venv/pytest_lib OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 nice -n 10 $PY -m pytest -q
```

Keep peft at 0.21.2: it rewrites a regex `target_modules` to bare suffixes for this MoE model type, and
`lora_exclude_modules` in `configs/models/qwen3_omni.yaml` is what keeps the audio tower out.
`test_peft_alone_would_widen_the_regex_to_the_audio_tower` fails if that behaviour changes, so re-check then.

## 2. Weights

The local Instruct snapshot is partial: config, index, preprocessor config, one 4.7 GB shard, no tokenizer. Finish it
(~70 GB; the talker and code2wav are in the same shards and are ignored on load). Run this on a node with a network and
a quota that allows it:

```bash
HF_HUB_CACHE=/fs/gamma-projects/audio/.cache/hub hf download Qwen/Qwen3-Omni-30B-A3B-Instruct
```

`configs/models/qwen3_omni.yaml` names the hub id, so set `HF_HUB_CACHE` and `HF_HUB_OFFLINE=1` at run time.

## 3. Smoke, on the local Captioner weights

```bash
sbatch scripts/slurm/smoke_qwen3.sbatch        # 2 rtxa6000, one rank across both, 2 h
```

It prints load time, step time, peak memory per GPU (after training, and with a worst-case 512-token probe), reward
values and whether the critic sync ran, and writes `smoke.json`. Read it knowing what it is:

- **Captioner weights, for local checks only.** Same class and sizes, but a downstream fine-tune for audio captions.
  It ignores the format prompt, so `--force-format` (on by default) swaps its paths for a fixed valid one, or no
  critic pass would run. Rewards say nothing about the method.
- **Real audio, fake video.** The local VGGSound tree has only WAVs, so each clip gets a generated test pattern.
- 128-token rollouts and 64-token critics, against PLAN's 512 and 256, to fit the 2 h job; the probe covers length.
- To try two real ranks: `sbatch --gres=gpu:rtxa6000:4 scripts/slurm/smoke_qwen3.sbatch`.

Then repeat with Instruct: `sbatch scripts/slurm/smoke_qwen3.sbatch --model-id <instruct snapshot>`.
Pass `--no-force-format` and check `malformed_rate`. That tells you whether θ_0 follows the format at all, which decides the
appendix's 20% rule. If `logprob_batch: 2` runs out of memory, set it to 1.

## 4. Data

**Quick setup.** `scripts/setup_datasets.sh <root>` fetches VGGSound, CMM and AVHBench, links AVQA clips from
VGGSound, and writes `<root>/datasets.env`. It stops on an audio-only VGGSound archive. `AVQA.zip` is a manual
download. Its real downloads are untested; only the logic ran, on stubs.

**AVQA.** Annotations (`train_qa.json`, `val_qa.json`) and a CSV of YouTube ids come from
[the dataset page](https://mn.cs.tsinghua.edu.cn/avqa). The page offers the videos only as that CSV, so clips are cut from
YouTube; PLAN's Baidu/OneDrive links may hold only annotations and features [VERIFY]. Lay out `<root>/train_qa.json`,
`<root>/val_qa.json`, `<root>/videos/<video_name>.mp4`. Rows with no clip are dropped and counted. Three guesses in
`avrsi/data/avqa.py` are marked [VERIFY]: list layout, 0-based `answer`, file naming. Check one row by eye. Confirm that
clips have audio; `load_audio` raises `WithholdingError` when a stream is missing.

**VGGSound with video.** Needed for XM, and **not on this filesystem**: the `video/` directory at
`naacl_audio/VGGSound` holds WAVs only (checked on 20k entries). SigLIP labels and swaps need frames, so XM cannot be built
from it. Get video clips first.

## 5. Build XM (Phase 0b, PLAN §5.2, §9)

`scripts/build_xm.py`, CPU apart from `plant`, in this order:

```bash
$PY scripts/build_xm.py similarity --classes classes.txt --out similarity.json
# Calibrate SigLIP on 300 human labels; measure precision on another 300 → threshold, class_precision.json.
# clips.csv columns: clip_id, source_video, label, siglip_score, partner_scores (JSON).
$PY scripts/build_xm.py splits --clips clips.csv --similarity similarity.json --threshold <t> \
    --class-precision class_precision.json --known-ids known_ids.txt --out xm_draft
$PY scripts/build_xm.py mux --xm-dir xm_draft --clip-paths clips.tsv --media-root xm_media
$PY scripts/build_xm.py plant --model qwen3_omni --xm-dir xm_draft --media-root xm_media --similarity similarity.json --out planted --split dev
# Iterate actor and critic prompts on planted XM-Dev items, then freeze them. Do not edit them after this line.
$PY scripts/build_xm.py plant --model qwen3_omni --xm-dir xm_draft --media-root xm_media --similarity similarity.json --out planted
# Human check of 500 items, then:
$PY scripts/build_xm.py freeze --xm-dir xm_draft --errors <n> --out xm_frozen
```

`plant` and `gate0.py` are single-process. On 48 GB cards set `gpus: [0, 1]` in `configs/models/qwen3_omni.yaml` for them.
`plant` writes the θ_0 paths the guard needs, so it runs on Instruct, never on the Captioner.

## 6. Gate 0 (Phase 1)

```bash
$PY scripts/gate0.py run --model qwen3_omni --planted planted/gate_planted.json --media-root xm_media --out gate0/qwen3_omni.json
$PY scripts/gate0.py decide gate0/*.json
```

Run the Qwen3 row first. PLAN needs two backbones to pass before training; only Qwen3 is prepared here, so AVF and
Qwen2.5 remain to do. A pilot without that is a systems check, not a green light.

## 7. Pilot (Phase 2)

`avrsi.train` refuses A8 without the guard: `--xm-dir` needs `train.jsonl`, `guard.jsonl`, `dev.jsonl`, and `--planted`
needs `guard_planted.json`. So the pilot cannot start before steps 4 to 6. `scripts/slurm/train_qwen3.sbatch` is a template for this:

```bash
ARM=A8 STEPS=1600 AVQA_ROOT=... XM_DIR=xm_frozen PLANTED=planted/guard_planted.json MEDIA_ROOT=xm_media sbatch scripts/slurm/train_qwen3.sbatch
ARM=A1 STEPS=1600 ... sbatch scripts/slurm/train_qwen3.sbatch
```

or directly:

```bash
torchrun --standalone --nproc_per_node=4 -m avrsi.train --arm A8 --model qwen3_omni --natural avqa --natural-root $AVQA \
    --xm-dir xm_frozen --planted planted/guard_planted.json --media-root xm_media --out outputs/pilot/A8 --steps 1600 --gpus-per-rank 1
```

Check: rewards are not constant, critic health stays out of saturation, length stays bounded, XM-Dev balanced accuracy moves.
`steps.jsonl` carries step time and critic health. `guard.jsonl` carries every decision. T=200 costs 1,792 guard passes,
shared among ranks. Add `--resume`, with the same rank count, to continue; the checkpoint interval is `dev_every` (400).

## Design notes

**Ranks.** Each holds a full model. `rng` draws the same global batch everywhere; a rank takes `batch[rank::world]`. Its K rollouts and
critic passes stay local. `Comm.reduce_grads` sums LoRA gradients, each rank having divided by the global trajectory count, so one
update equals a single process on the whole batch (tested). AdamW then runs identically; digests of the actor are compared at each
checkpoint and a mismatch raises. Guard and dev scoring are split among ranks and gathered, so every rank decides from the same scores and
makes the same critic sync. Rank 0 alone writes logs and checkpoints; per-rank RNG streams go in `state.pt`. Self-preference runs on rank 0.

**Memory for scoring.** Rollouts and critic passes run under `no_grad`. Each prompt's K trajectories are scored in forwards of
`logprob_batch` (default 2), right-padded, each backpropagated and freed before the next, with reference log-probabilities computed
first. A prompt is ~2.4k tokens (2,200 video + 130 audio at 640x360, 10 s, 20 frames). Non-reentrant gradient checkpointing covers the decoder only, as the
towers have no trainable weights and stay in eval mode. `lm_head` sees only the completion, saving ~1 GB per trajectory. Towers do rerun
per forward; sharing their output would need bypassing the thinker's `forward`.

**Dtypes.** Base bf16, LoRA float32 (PEFT's autocast, kept on purpose: bf16's 8 mantissa bits would round a 1e-5 AdamW step away).
Log-probabilities are computed in float32, since bf16 logits are too coarse for the KL.

**EOS.** The thinker's configs have no EOS; only the composite class passes one. `_generate_kwargs` supplies `<|im_end|>` and
`<|endoftext|>`. Without them every rollout runs to `max_new_tokens` and fails ValidFormat.

**`num_frames: 20`.** `sample_frames` refuses `num_frames` with `fps`, while the processor reads `fps` (default 1) to place video
against audio. So `sampled_video_kwargs` asks for the clip's own rate N/duration with `do_sample_frames=True`, and frames are decoded
by PyAV and cached.

**LoRA.** 192 modules: q/k/v/o in 48 decoder layers, 26,738,688 trainable parameters at r=32. Experts and the router are raw
`nn.Parameter`s, so PEFT cannot wrap them as modules. `lora_alpha` stays PEFT's default 8, scale 0.25; PLAN does not fix it, and it
scales the effective learning rate, so decide it before the pilot.

## Unverified until a GPU run

- That real weights load this way on a GPU: time, host RAM during expert fusion, and whether `--mem=120G` is enough. Keys and
  shapes match the checkpoint and fusion is tested on a tiny checkpoint in the hub layout.
- All GPU memory numbers, including the ~72 GB estimate and `logprob_batch: 2`.
- Step and guard time. The 1,792 guard passes per check and 112 critic passes per step are counts, not durations.
- NCCL. Tests use gloo on CPU. Bit-identical AdamW steps across GPUs are checked at run time, not proven.
- A rank spanning GPUs: accelerate hooks with PEFT, checkpointing and the `lm_head` crop have run on one CPU device only. The map is
  computed on the meta device and fits.
- Whether Instruct follows the format, and the quality of its critic verdicts (Gate 0).
- AVQA file layout, marked [VERIFY] in `avrsi/data/avqa.py`; the videos' audio tracks.
- The `use_audio_in_video` interleaving is checked on token counts and positions, not on model behaviour.
