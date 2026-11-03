# AVRSI

Recursively self-improving audio-visual reasoners with cross-modal checking: GRPO training where the
critic checks audio claims with video withheld and visual claims with audio withheld, and is refreshed
from the improving policy behind a guard. The design is in [docs/PLAN.md](docs/PLAN.md); its appendix
overrides the body.

## Status

**One real-weights smoke run (2 steps of A8 on 2 RTX A6000s, docs/RUN_QWEN3.md "Measured"); no pilot yet.** The reward, critic, guard, GRPO, XM, metrics and statistics code is unit tested on a
fake backend. Qwen3-Omni runs on the real thinker class (tiny random config), real PEFT and the real processor on a generated
MP4, with training, resume and two-process runs tested on CPU; see [docs/RUN_QWEN3.md](docs/RUN_QWEN3.md), which also lists what
only a GPU run can settle. The thinker needs 63.4 GB of GPU memory. The Qwen2.5-Omni and AVF wrappers remain unexecuted, and Phase 0
`[VERIFY]` items other than Qwen3's are open. Critic and actor prompts are drafts until frozen on planted XM-Dev items.

## Layout

See PLAN §11: `avrsi/` (library), `configs/` (ablations and per-model settings as data), `scripts/`
(`build_xm.py`, `gate0.py`, `evaluate.py`, `audit.py`), `tests/`.

`python -m avrsi.config --print-runs` lists a backbone's runs from `run_matrix`; `--select-tuned` applies §5.3.
`python -m avrsi.train` takes `--T` for A14 and `--resume` to continue from the checkpoint written every
`dev_every` steps. Launch it through `torchrun -m avrsi.train` for several GPUs. `scripts/smoke_qwen3.py` and
`scripts/slurm/` hold the real-weights smoke run and job templates. `build_xm.py` order: `splits`, `mux`, `plant --split dev`, prompt freeze, `plant`, human
check, `freeze`.

## Tests

```bash
pip install -e '.[test]'
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 nice -n 10 python -m pytest
```

No GPU, data or network is needed. Judge tests use a mocked transport. Qwen3 tests need `.venv/run_lib`
(`requirements-qwen3.txt`) and skip without it or without the local snapshots.

## Judges

Keys come from the env vars or key file named in `configs/experiment.yaml`, go only in request
headers, and are never logged. `scripts/audit.py` only prints the planned calls unless given `--execute`.

## Licence note

AVF is under the NVIDIA OneWay Noncommercial licence. Research use only.
