"""Ablations and per-model settings as data (PLAN §8, §4). configs/ is the source of truth."""
from __future__ import annotations

from dataclasses import dataclass, fields, replace
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
CONFIGS = ROOT / "configs"
T_GRID = (100, 200, 500)
TUNABLE = ("alpha", "gamma", "T", "p")


@dataclass(frozen=True)
class Arm:
    id: str
    K: int
    B: int
    p: float
    T: int | None  # None = ∞
    T_max: int
    lam: float
    alpha: float
    gamma: float
    beta: float
    lr: float
    lora_r: int
    max_new_tokens: int
    dev_every: int
    pair_orders: str
    check_view: str
    critic: str
    guard: str
    guard_view: str
    reward: str
    trained: bool
    retired: bool
    t_sweep: bool

    @property
    def has_handoff(self) -> bool:
        return self.guard != "none"


def load_experiment(path: Path | str | None = None) -> dict:
    with open(path or CONFIGS / "experiment.yaml") as f:
        return yaml.safe_load(f)


def load_model(name: str, path: Path | str | None = None) -> dict:
    with open(path or CONFIGS / "models" / f"{name}.yaml") as f:
        return yaml.safe_load(f)


def resolve_arm(arm_id: str, tuned: dict | None = None, experiment: dict | None = None) -> Arm:
    """Tuned values replace defaults for every system; an arm's own change then wins (§8)."""
    exp = experiment or load_experiment()
    tuned = tuned or {}
    unknown = set(tuned) - set(TUNABLE)
    if unknown:
        raise ValueError(f"not tunable: {sorted(unknown)}")
    values = {**exp["defaults"], **tuned, **exp["arms"][arm_id], "id": arm_id}
    if arm_id == "A12" and tuned.get("p") == 0:  # §8: if tuning selects p=0, A12 runs p=0.3
        values["p"] = 0.3
    allowed = {f.name for f in fields(Arm)}
    return Arm(**{k: v for k, v in values.items() if k in allowed})


def expand_arm(arm_id: str, tuned: dict | None = None, experiment: dict | None = None) -> list[Arm]:
    """Runnable systems of an ablation: A14 is one per T not chosen for A8, A13 is none."""
    arm = resolve_arm(arm_id, tuned, experiment)
    if arm.retired:
        return []
    if arm.t_sweep:
        return [replace(arm, T=t) for t in T_GRID if t != arm.T]
    return [arm]


def backbone_arms(backbone: str, gate0_passed: bool, experiment: dict | None = None) -> list[str]:
    """§8 and §9: which systems run on a backbone."""
    exp = experiment or load_experiment()
    matrix = exp["run_matrix"]
    if not gate0_passed:
        return ["A0"]
    arms = matrix["full"] if backbone == matrix["primary_backbone"] else matrix["cross_backbone"]
    return ["A0", *arms]


def training_runs(backbone: str, gate0_passed: bool, experiment: dict | None = None) -> list[tuple[Arm, int]]:
    """(arm, seed) per full training run (§7.4). A0 is evaluation only."""
    exp = experiment or load_experiment()
    matrix = exp["run_matrix"]
    if not gate0_passed:
        return []
    arms = matrix["full"] if backbone == matrix["primary_backbone"] else matrix["cross_backbone"]
    return [(arm, seed) for arm_id, seeds in arms.items() for arm in expand_arm(arm_id, experiment=exp) for seed in range(seeds)]


def tuning_values(experiment: dict | None = None) -> list[dict]:
    """§5.3: the default, then one-factor-at-a-time off-default values."""
    exp = experiment or load_experiment()
    grid, defaults = exp["tuning"], exp["defaults"]
    runs = [{}]
    for key in TUNABLE:
        runs += [{key: v} for v in grid[key] if v != defaults[key]]
    return runs


def short_runs(experiment: dict | None = None) -> list[dict]:
    """Pilot plus tuning runs at one seed each, 1,600 steps (§6)."""
    exp = experiment or load_experiment()
    pilot = [{"arm": a, "steps": exp["run_matrix"]["pilot"]["steps"]} for a in exp["run_matrix"]["pilot"]["arms"]]
    return pilot + [{"arm": "A8", "steps": exp["tuning"]["steps"], "tuned": t} for t in tuning_values(exp)]
