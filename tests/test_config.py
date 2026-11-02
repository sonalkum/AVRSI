import json
from dataclasses import asdict

import pytest

from avrsi.config import (backbone_arms, expand_arm, load_experiment, load_model, resolve_arm, select_arm, select_tuned, short_runs, training_runs,
                          tuning_values)
from avrsi.config import main as config_main

A8 = asdict(resolve_arm("A8"))
NO_HANDOFF = {"guard": "none", "guard_view": "none"}
# Written by hand from PLAN §8, so a config edit cannot pass by matching itself.
EXPECTED = {
    "A0": {"trained": False, "critic": "none", "pair_orders": "none", "check_view": "none", **NO_HANDOFF},
    "A1": {"alpha": 0.0, "gamma": 0.0, "critic": "none", "pair_orders": "none", "check_view": "none", **NO_HANDOFF},
    "A2": {"gamma": 0.0, "check_view": "none", "guard_view": "joint"},
    "A3": {"critic": "gemini", "check_view": "joint", **NO_HANDOFF},
    "A4": {"critic": "gemini", **NO_HANDOFF},
    "A5": {"T": None, **NO_HANDOFF},
    "A6": {"alpha": 0.0, "pair_orders": "none"},
    "A7": {"reward": "rl_comm", "alpha": 0.0, "gamma": 0.0, "critic": "lalm", "pair_orders": "none", "check_view": "none", **NO_HANDOFF},
    "A8": {},
    "A9": {"pair_orders": "both"},
    "A10": {"guard": "minibatch", "guard_view": "none"},
    "A11": {"K": 8},
    "A12": {"p": 0.0},
    "A13": {"retired": True},
    "A14": {"t_sweep": True},
    "A15": {"check_view": "joint", "guard_view": "joint"},
}


def test_all_ablations_present():
    assert set(load_experiment()["arms"]) == {f"A{i}" for i in range(16)}


@pytest.mark.parametrize("arm_id", sorted(EXPECTED))
def test_each_ablation_differs_from_a8_only_as_plan_says(arm_id):
    arm = asdict(resolve_arm(arm_id))
    diff = {k: v for k, v in arm.items() if k != "id" and v != A8[k]}
    assert diff == EXPECTED[arm_id]


def test_a8_defaults_match_plan():
    arm = resolve_arm("A8")
    expected = dict(K=4, B=8, p=0.3, T=200, T_max=6400, lam=1.0, alpha=1.0, gamma=1.0, beta=0.01, lr=1e-5, lora_r=32, max_new_tokens=512,
                    dev_every=400, pair_orders="single", check_view="isolated", critic="self", guard="xm_guard", guard_view="isolated")
    assert {k: getattr(arm, k) for k in expected} == expected


def test_sampling_config():
    s = load_experiment()["sampling"]
    assert s["rollout"] == {"temperature": 1.0, "top_p": 1.0, "do_sample": True, "top_k": 0, "repetition_penalty": 1.0}
    assert s["greedy"]["do_sample"] is False and s["auroc_sampled_passes"] == 4


def test_no_handoff_set():
    no = {a for a in EXPECTED if not resolve_arm(a).has_handoff}
    assert no == {"A0", "A1", "A3", "A4", "A5", "A7"}


def test_tuned_values_replace_defaults_for_every_system_but_arm_changes_win():
    tuned = {"alpha": 0.5, "gamma": 2.0, "T": 100, "p": 0.0}
    assert resolve_arm("A8", tuned).alpha == 0.5
    assert resolve_arm("A15", tuned).T == 100
    assert resolve_arm("A1", tuned).alpha == 0.0 and resolve_arm("A1", tuned).p == 0.0
    assert resolve_arm("A5", tuned).T is None


def test_a12_flips_when_tuning_selects_p0():
    assert resolve_arm("A12").p == 0.0
    assert resolve_arm("A12", {"p": 0.0}).p == 0.3


def test_a14_runs_unchosen_t():
    assert [a.T for a in expand_arm("A14")] == [100, 500]
    assert [a.T for a in expand_arm("A14", {"T": 100})] == [200, 500]


def test_a13_never_runs():
    assert expand_arm("A13") == []


def test_unknown_tuned_key_rejected():
    with pytest.raises(ValueError):
        resolve_arm("A8", {"beta": 1})


def test_backbone_sets():
    assert backbone_arms("avf", True) == ["A0", "A1", "A2", "A8", "A15"]
    assert backbone_arms("avf", False) == ["A0"]
    assert "A5" in backbone_arms("qwen3_omni", True)


def test_run_counts_section_7_4():
    assert len(training_runs("qwen3_omni", True)) == 25
    assert len(training_runs("avf", True)) == 12
    assert len(training_runs("avf", False)) == 0
    assert len(training_runs("qwen3_omni", True)) + 2 * len(training_runs("avf", True)) == 49
    assert len(short_runs()) == 9
    assert len(tuning_values()) == 7


@pytest.mark.parametrize("name,flag,frames", [("qwen3_omni", "use_audio_in_video", 20), ("qwen2_5_omni", "use_audio_in_video", None),
                                              ("avf", "load_audio_in_video", 128)])
def test_model_settings(name, flag, frames):
    config = load_model(name)
    assert config["backend"] == name and config["audio_in_video_flag"] == flag and config["num_frames"] == frames
    assert config["lora_target_modules"] is None  # [VERIFY] until Phase 0


def test_avf_settings():
    config = load_model("avf")
    assert (config["image_size"], config["audio_window_s"]) == (448, 30) and "Noncommercial" in config["licence"]




def test_a14_T_list_excludes_the_tuned_T():
    assert [a.T for a in expand_arm("A14", {"T": 100})] == [200, 500]
    runs = [arm.T for arm, _ in training_runs("qwen3_omni", True, tuned={"T": 500}) if arm.id == "A14"]
    assert runs == [100, 200]
    a8 = {arm.T for arm, _ in training_runs("qwen3_omni", True, tuned={"T": 500}) if arm.id == "A8"}
    assert a8 == {500}


def test_select_arm_rules():
    assert select_arm("A14", {"T": 100}, 500).T == 500
    assert select_arm("A8").id == "A8"
    with pytest.raises(ValueError, match="needs --T"):
        select_arm("A14")
    with pytest.raises(ValueError, match="T=200"):
        select_arm("A14", None, 200)
    with pytest.raises(ValueError, match="retired"):
        select_arm("A13")


def test_evaluation_only_comes_from_the_matrix():
    experiment = load_experiment()
    experiment["run_matrix"]["evaluation_only"] = ["A0", "A99"]
    assert backbone_arms("avf", False, experiment) == ["A0", "A99"]
    assert backbone_arms("avf", True, experiment)[:2] == ["A0", "A99"]


def runs(**scores):
    rows = [{"tuned": {}, "score": 0.70}]
    return rows + [{"tuned": {k: v}, "score": s} for k, (v, s) in scores.items()]


def test_tuning_keeps_defaults_under_one_point():
    assert select_tuned(runs(alpha=(0.5, 0.7099), gamma=(2.0, 0.70))) == {}


def test_tuning_takes_a_gain_of_one_point():
    assert select_tuned(runs(alpha=(0.5, 0.71), T=(100, 0.75))) == {"alpha": 0.5, "T": 100}


def test_tuning_picks_best_value_per_factor():
    rows = runs() + [{"tuned": {"gamma": 0.5}, "score": 0.72}, {"tuned": {"gamma": 2.0}, "score": 0.74}]
    assert select_tuned(rows) == {"gamma": 2.0}


def test_tuning_needs_the_default_run():
    with pytest.raises(ValueError):
        select_tuned([{"tuned": {"T": 100}, "score": 0.7}])


def test_print_runs(capsys):
    config_main(["--print-runs", "--backbone", "avf"])
    lines = capsys.readouterr().out.splitlines()
    assert lines[0] == "avf A0 evaluation_only" and len(lines) == 13
    assert "avf A15 T=200 seed=2 steps=6400" in lines
    config_main(["--print-runs", "--tuned", '{"T": 100}'])
    assert "qwen3_omni A14 T=500 seed=0 steps=6400" in capsys.readouterr().out
    config_main(["--print-runs", "--gate0-failed"])
    assert capsys.readouterr().out.splitlines() == ["qwen3_omni A0 evaluation_only"]


def test_select_tuned_cli(tmp_path, capsys):
    path = tmp_path / "r.json"
    path.write_text(json.dumps(runs(T=(500, 0.8))))
    config_main(["--select-tuned", str(path)])
    assert json.loads(capsys.readouterr().out) == {"T": 500}
