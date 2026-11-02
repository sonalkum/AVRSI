import numpy as np
import pytest

from avrsi.stats import (H5_MARGIN, Term, accuracy_term, avhbench_term, cmm_term, connected_components, decide, decide_backbone, holm,
                         intersection_union, mcnemar, nested_recall, paired_delta_bootstrap, xm_term)


def test_connected_components_transitive():
    assert connected_components([("a", "b"), ("b", "c"), ("d",), ("c",), ("e", "d")]) == [0, 0, 1, 0, 1]


def test_nested_recall_family_then_class():
    correct = np.array([1, 1, 0, 1], float)
    outer, inner = np.array([0, 0, 0, 1]), np.array([0, 0, 1, 0])
    assert nested_recall(correct, outer, inner) == 0.75


def test_nested_recall_skips_lost_cells():
    assert nested_recall(np.array([1.0]), np.array([1]), np.array([1])) == 1.0


def test_term_matches_metrics_xm():
    from avrsi.metrics import xm_balanced_accuracy
    fam, cls, ok = ["F1", "F1", "F1", "F4", "F4"], ["yes", "yes", "no", "both", "neither"], [1, 1, 0, 1, 0]
    term = xm_term([(f"s{i}",) for i in range(5)], [f"t{i}" for i in range(5)], fam, cls)
    rows = [dict(family=f, answer_class=c, correct=bool(o)) for f, c, o in zip(fam, cls, ok)]
    assert term.score(np.array([ok], float)) == pytest.approx(xm_balanced_accuracy(rows))


def test_term_matches_metrics_cmm_and_avh():
    from avrsi.metrics import avhbench_score, cmm_score
    golds = ["yes", "yes", "no", "no"]
    preds = ["yes", "no", "no", "yes"]
    ok = np.array([[p == g for p, g in zip(preds, golds)]], float)
    assert cmm_term([0, 0, 1, 1], golds).score(ok) == cmm_score(preds, golds)
    tasks = ["a2v", "v2a", "a2v", "v2a"]
    rows = [dict(task=t, pred=p, gold=g) for t, p, g in zip(tasks, preds, golds)]
    assert avhbench_term(tasks, golds).score(ok) == avhbench_score(rows)


def test_cmm_cluster_is_the_sample():
    term = cmm_term([0, 0, 1, 1], ["yes", "no", "yes", "no"])
    assert term.clusters.tolist() == [0, 0, 1, 1]


def test_xm_cluster_is_connected_component_of_reused_clips_and_twins():
    term = xm_term([("a",), ("a", "b"), ("c",), ("b",), ("d",)], ["t1", "t1", "t2", "t3", "t2"], ["F1"] * 5, ["yes"] * 5)
    assert term.clusters.tolist() == [0, 0, 1, 0, 1]


def test_avhbench_cluster_is_the_item():
    assert avhbench_term(["a2v"] * 3, ["yes"] * 3).clusters.tolist() == [0, 1, 2]


def boot(a, b, clusters=None, B=2000):
    n = a.shape[-1]
    term = accuracy_term(clusters, n)
    return paired_delta_bootstrap({"x": term}, {"x": np.atleast_2d(a)}, {"x": np.atleast_2d(b)}, B=B, seed=0)


def test_identical_systems_delta_zero():
    a = np.random.default_rng(0).integers(0, 2, 200).astype(float)
    result = boot(a, a, B=200)
    assert result["delta"] == 0 and result["lower"] == 0 and result["p"] == 1.0


def test_pairing_tightens_the_bound():
    # B is A with 5% more correct: paired variance is tiny even though scores are noisy.
    rng = np.random.default_rng(0)
    b = (rng.random(1000) < 0.5).astype(float)
    a = b.copy()
    wrong = np.flatnonzero(b == 0)
    a[wrong[:50]] = 1
    result = boot(a, b)
    assert result["delta"] == pytest.approx(0.05) and result["lower"] > 0.03


def test_clustering_widens_the_interval():
    # Each cluster repeats one outcome 10 times. Item-level resampling would be far too narrow.
    rng = np.random.default_rng(1)
    diff = rng.choice([-1.0, 0.0, 1.0], 20).repeat(10)
    a, b = np.clip(diff, 0, 1), np.clip(-diff, 0, 1)
    item_level = boot(a, b, None)
    clustered = boot(a, b, np.arange(20).repeat(10))
    assert clustered["delta"] == item_level["delta"]
    assert (clustered["delta"] - clustered["lower"]) > 2 * (item_level["delta"] - item_level["lower"])


def test_seeds_are_averaged_and_counts_may_differ():
    n = 100
    term = accuracy_term(None, n)
    a = np.vstack([np.ones(n), np.zeros(n), np.ones(n)])
    b = np.full((1, n), 0.5)
    assert paired_delta_bootstrap({"x": term}, {"x": a}, {"x": b}, B=10)["delta"] == pytest.approx(2 / 3 - 0.5)


def test_composite_is_mean_of_term_deltas_stratified():
    x, y = accuracy_term(None, 50), accuracy_term(None, 500)
    ones, zeros = np.ones((1, 50)), np.zeros((1, 50))
    result = paired_delta_bootstrap({"x": x, "y": y}, {"x": ones, "y": np.ones((1, 500))}, {"x": zeros, "y": np.ones((1, 500))}, B=20)
    assert result["delta"] == 0.5  # unweighted by benchmark size


def test_one_sided_lower_is_5th_percentile():
    term = Term(np.arange(4), np.zeros(4, int), np.zeros(4, int))
    a = np.array([[1, 1, 1, 0]], float)
    b = np.zeros((1, 4))
    result = paired_delta_bootstrap({"x": term}, {"x": a}, {"x": b}, B=4000, seed=1)
    assert result["lower"] == 0.25  # P(X≤0)=.004, P(X≤1)=.051 for Binomial(4, .75)
    assert result["delta"] == 0.75


def test_seed_range_reported():
    n = 10
    term = accuracy_term(None, n)
    a = np.vstack([np.ones(n), np.zeros(n)])
    result = paired_delta_bootstrap({"x": term}, {"x": a}, {"x": np.zeros((1, n))}, B=5)
    assert result["seed_range_a"] == [0.0, 1.0] and result["seed_range_b"] == [0.0, 0.0]


def test_holm_step_down():
    assert holm({"a": 0.01, "b": 0.04, "c": 0.03}) == {"a": True, "b": False, "c": False}
    assert holm({"a": 0.01, "b": 0.02, "c": 0.04}) == {"a": True, "b": True, "c": True}
    assert holm({"a": 0.06, "b": 0.001}) == {"a": False, "b": True}


def test_holm_stops_at_first_failure():
    assert holm({"a": 0.02, "b": 0.03, "c": 0.04, "d": 0.045, "e": 0.05}) == {k: False for k in "abcde"}


def test_iu_requires_every_backbone():
    assert intersection_union([0.01, 0.02])
    assert not intersection_union([0.01, 0.0])
    assert not intersection_union([])


def test_h5_margin_is_one_point():
    assert H5_MARGIN == -0.01
    assert intersection_union([-0.009], H5_MARGIN) and not intersection_union([-0.011], H5_MARGIN)


def test_mcnemar():
    a = [True] * 10 + [False] * 10
    assert mcnemar(a, a) == 1.0
    b = [False] * 10 + [False] * 10
    assert mcnemar(a, b) == pytest.approx(2 * 0.5 ** 10)
    assert mcnemar([True, False], [False, True]) == 1.0


def contrast(lower, p):
    return {"delta": 0.02, "lower": lower, "p": p}


def test_decide_backbone_needs_bound_and_holm():
    results = {"H3": contrast(0.01, 0.001), "H3b": contrast(0.01, 0.04), "H4_A5": contrast(-0.1, 0.002), "H4_A4": contrast(0.1, 0.03), "H4_A3": contrast(0.1, 0.5)}
    assert decide_backbone(results) == {"H3": True, "H3b": False, "H4_A5": False, "H4_A4": False, "H4_A3": False}


def test_decide_full():
    ok = {"H1": contrast(0.01, 0.01), "H5": contrast(-0.005, 0.3), "H3": contrast(0.01, 0.01), "H3b": contrast(0.01, 0.01)}
    bad = {**ok, "H1": contrast(-0.01, 0.2), "H5": contrast(-0.02, 0.5)}
    assert decide({"a": ok, "b": ok})["H1"] and decide({"a": ok, "b": ok})["H5"]
    assert not decide({"a": ok, "b": bad})["H1"] and not decide({"a": ok, "b": bad})["H5"]
    assert decide({"a": ok})["per_backbone"]["a"] == {"H3": True, "H3b": True}


def test_lower_bound_is_the_one_sided_quantile():
    a = np.array([[1, 1, 0, 1, 1, 0, 1, 1, 1, 0] * 3], float)
    b = np.array([[1, 0, 0, 1, 0, 0, 1, 0, 1, 0] * 3], float)
    n = a.shape[1]
    term = accuracy_term(None, n)
    B = 400
    rng = np.random.default_rng(3)
    boot = [a[0][p].mean() - b[0][p].mean() for p in (rng.integers(n, size=n) for _ in range(B))]
    out = paired_delta_bootstrap({"t": term}, {"t": a}, {"t": b}, B=B, seed=3)
    assert out["lower"] == pytest.approx(float(np.quantile(boot, 0.05)))
