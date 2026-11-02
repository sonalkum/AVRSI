# AVRSI

Recursively self-improving audio-visual reasoners with cross-modal checking: GRPO training where the
critic checks audio claims with video withheld and visual claims with audio withheld, and is refreshed
from the improving policy behind a guard. The design is in [docs/PLAN.md](docs/PLAN.md); its appendix
overrides the body.

## Status

**Untested on real models.** No GPU, dataset or live judge API has been used. The reward, critic,
guard, GRPO, XM, metrics and statistics code is unit tested on a fake backend. The Qwen3-Omni,
Qwen2.5-Omni and AVF wrappers and the `scripts/` model paths have never been executed, and every
Phase 0 `[VERIFY]` item is open. Critic and actor prompts are drafts until frozen on XM-Dev.

## Layout

See PLAN §11: `avrsi/` (library), `configs/` (ablations and per-model settings as data), `scripts/`
(`build_xm.py`, `gate0.py`, `evaluate.py`, `audit.py`), `tests/`.

## Tests

```bash
pip install -e '.[test]'
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 nice -n 10 python -m pytest
```

No GPU, data or network is needed. Judge tests use a mocked transport.

## Judges

Keys come from the env vars or key file named in `configs/experiment.yaml`, go only in request
headers, and are never logged. `scripts/audit.py` only prints the planned calls unless given `--execute`.

## Licence note

AVF is under the NVIDIA OneWay Noncommercial licence. Research use only.
