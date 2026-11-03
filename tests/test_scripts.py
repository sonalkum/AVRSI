import importlib.util
import json
import random
from pathlib import Path

import pytest
from conftest import make_disjoint
from test_sync import BASE

from avrsi.models.base import Generation
from avrsi.models.fake import FakeBackend
from avrsi.perturb import freeze, load_draft, load_frozen, load_planted, write_draft

ROOT = Path(__file__).resolve().parent.parent
ANSWER = "<audio>A sound.</audio><visual>A view.</visual><fusion></fusion><answer>{}</answer>"


def load(name):
    spec = importlib.util.spec_from_file_location(f"script_{name}", ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("name", ["gate0", "build_xm", "evaluate", "audit"])
def test_help(name, capsys):
    with pytest.raises(SystemExit) as stop:
        load(name).main(["--help"])
    assert stop.value.code == 0 and "usage" in capsys.readouterr().out


def splits_argv(tmp_path, precision=None):
    tmp_path.mkdir(exist_ok=True)
    classes = ["dog", "piano", "rain", "siren", "cat"]
    rows = ["clip_id,source_video,label,siglip_score,partner_scores"]
    partners = json.dumps(dict.fromkeys(classes, 0.1)).replace('"', '""')
    rows += [f'c{i},s{i},{classes[i % 5]},0.9,"{partners}"' for i in range(500)]
    csv = tmp_path / "clips.csv"
    csv.write_text("\n".join(rows))
    sim = tmp_path / "sim.json"
    sim.write_text(json.dumps({f"{a}|{b}": 0.1 for i, a in enumerate(classes) for b in classes[i + 1:]}))
    known = tmp_path / "known.txt"
    known.write_text("elsewhere\n")
    table = tmp_path / "precision.json"
    table.write_text(json.dumps(precision or dict.fromkeys(classes, 0.95)))
    out = tmp_path / "xm"
    return out, ["splits", "--clips", str(csv), "--similarity", str(sim), "--threshold", "0.5", "--class-precision", str(table),
                 "--known-ids", str(known), "--out", str(out)]


@pytest.fixture
def frozen(tmp_path, xm):
    path = tmp_path / "test.jsonl"
    freeze(xm["dev"], path)
    return path, xm["dev"]


def write_preds(path, items, answer):
    path.write_text("\n".join(json.dumps({"item_id": i.item_id, "text": ANSWER.format(answer(i)), "tokens": 5}) for i in items))


def test_evaluate_score_xm(frozen, tmp_path, capsys):
    path, items = frozen
    preds = tmp_path / "p.jsonl"
    write_preds(preds, items, lambda i: i.answer)
    load("evaluate").main(["score", "--benchmark", "xm", "--annotations", str(path), "--predictions", str(preds)])
    out = json.loads(capsys.readouterr().out)
    assert out["xm_balanced_accuracy"] == 1.0 and out["swap_sensitivity"] == 1.0


@pytest.fixture
def xrs_args(frozen, tmp_path):
    """ΔXRS inputs: perfect system a, always-no system b, on all three benchmarks."""
    path, items = frozen
    cmm = tmp_path / "cmm.json"
    cmm.write_text(json.dumps([{"sub_category": "visual-language", "answer": "yes" if i % 2 else "no", "question": f"q{i}", "video_path": "v.mp4",
                                "audio_path": ""} for i in range(40)]))
    avh = tmp_path / "avh.json"
    avh.write_text(json.dumps([{"video_id": f"v{i}", "task": "Audio-driven Video Hallucination" if i % 2 else "Video-driven Audio Hallucination",
                                "text": "q", "label": "Yes" if i % 4 < 2 else "No"} for i in range(40)]))
    golds = {"xm": items, "cmm": load("evaluate").load_benchmark("cmm", cmm), "avhbench": load("evaluate").load_benchmark("avhbench", avh)}
    specs = {}
    for name, rows in golds.items():
        a, b = tmp_path / f"{name}_a.jsonl", tmp_path / f"{name}_b.jsonl"
        write_preds(a, rows, lambda i: i.answer)
        write_preds(b, rows, lambda i: "no" if not i.options else i.options[0])
        specs[name] = (a, b)
    paths = {"xm": path, "cmm": cmm, "avhbench": avh}
    return specs, ["--annotation", *[f"{n}={p}" for n, p in paths.items()], "--media-root", "xm=", "cmm=", "avhbench=", "--bootstraps", "30"]


def compare_argv(specs, tail, names=None, b_names=None):
    names = names or list(specs)
    return ["compare", "--a", *[f"{n}={specs[n][0]}" for n in names], "--b", *[f"{n}={specs[n][1]}" for n in (b_names or names)], *tail]


def test_evaluate_compare(xrs_args, capsys):
    specs, tail = xrs_args
    load("evaluate").main(compare_argv(specs, tail))
    out = json.loads(capsys.readouterr().out)
    assert out["delta"] > 0.3 and out["lower"] > 0 and out["xrs_a"] == 1.0
    assert out["xrs_a"] - out["xrs_b"] == pytest.approx(out["delta"])


def test_compare_refuses_partial_xrs(xrs_args):
    specs, tail = xrs_args
    with pytest.raises(SystemExit, match="ΔXRS needs"):
        load("evaluate").main(compare_argv(specs, tail, names=["xm"]))


def test_compare_refuses_mismatched_b(xrs_args):
    specs, tail = xrs_args
    with pytest.raises(SystemExit, match="same benchmarks"):
        load("evaluate").main(compare_argv(specs, tail, b_names=["xm", "xm", "cmm"]))


def test_compare_accuracy_mode_needs_no_xrs_terms(xrs_args, capsys):
    specs, tail = xrs_args
    load("evaluate").main(compare_argv(specs, [*tail, "--accuracy"], names=["cmm"]))
    assert "xrs_a" not in json.loads(capsys.readouterr().out)


def test_predict_modalities_and_digest(tmp_path, xm, monkeypatch):
    from avrsi.critic import prompt_digest
    from avrsi.models.fake import FakeBackend

    backend = FakeBackend()
    monkeypatch.setattr("avrsi.models.load_backend", lambda model: backend)
    path = tmp_path / "dev.jsonl"
    freeze(xm["dev"][:6], path)
    out = tmp_path / "p.jsonl"
    argv = ["predict", "--benchmark", "xm", "--annotations", str(path), "--root", "m", "--model", "qwen3_omni", "--out", str(out)]
    load("evaluate").main([*argv, "--modalities", "video"])
    assert all(x.video and x.audio is None for _, x in backend.seen_inputs) and len(backend.seen_inputs) == 6
    assert json.loads(Path(f"{out}.meta.json").read_text())["prompt_digest"] == prompt_digest()
    backend.seen_inputs.clear()
    load("evaluate").main(argv)
    assert all(x.video and x.audio for _, x in backend.seen_inputs)


def test_evaluate_decide(tmp_path, capsys):
    ok = {"delta": 0.02, "lower": 0.01, "p": 0.01}
    results = tmp_path / "r.json"
    results.write_text(json.dumps({"qwen3_omni": {"H1": ok, "H5": ok, "H3": ok, "H3b": ok}}))
    load("evaluate").main(["decide", "--results", str(results)])
    assert json.loads(capsys.readouterr().out)["H1"] is True


def test_audit_plans_without_calling(frozen, tmp_path, capsys, monkeypatch):
    path, items = frozen
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    a = tmp_path / "a.jsonl"
    write_preds(a, items, lambda i: i.answer)
    load("audit").main(["--items", str(path), "--media-root", "", "--ours", str(a), "--baseline", str(a), "--out", str(tmp_path / "o")])
    assert "planned: 512 items" in capsys.readouterr().out
    assert not (tmp_path / "o").exists()


def test_build_xm_splits(tmp_path, capsys):
    out, argv = splits_argv(tmp_path)
    load("build_xm").main(argv)
    assert not (out / "test.jsonl.sha256").exists()  # frozen only after the human check
    assert len((out / "gate.jsonl").read_text().splitlines()) == 500
    assert len(load_draft(out / "gate.jsonl")) == 500


def test_freeze_needs_the_human_check(tmp_path):
    drafts, argv = splits_argv(tmp_path)
    load("build_xm").main(argv)
    frozen_dir = tmp_path / "frozen"
    freeze_argv = ["freeze", "--xm-dir", str(drafts), "--out", str(frozen_dir), "--checked", "500"]
    with pytest.raises(SystemExit, match="human check failed"):
        load("build_xm").main([*freeze_argv, "--errors", "26"])
    assert not frozen_dir.exists()
    load("build_xm").main([*freeze_argv, "--errors", "25"])
    assert len(load_frozen(frozen_dir / "test.jsonl")) == 5000 and (frozen_dir / "train.jsonl.sha256").exists()


def test_splits_wires_precision_shares_and_requires_inputs(tmp_path, capsys):
    out, argv = splits_argv(tmp_path, precision={"dog": 0.5, "piano": 0.95, "rain": 0.95, "siren": 0.9, "cat": 0.9})
    load("build_xm").main(argv)
    assert "excluded 1 classes" in capsys.readouterr().out
    assert "dog" not in {i.visual_label for i in load_draft(out / "dev.jsonl")}
    with pytest.raises(SystemExit):
        load("build_xm").main([a for a in argv if a != "--known-ids"])
    out, argv = splits_argv(tmp_path / "again", precision={"dog": 0.95})
    with pytest.raises(SystemExit, match="no measured precision"):
        load("build_xm").main(argv)


def test_splits_family_shares_come_from_config(tmp_path, monkeypatch):
    module = load("build_xm")
    experiment = module.load_experiment()
    experiment["xm"]["family_shares"] = {"F1": 0.5, "F2": 0.5, "F3": 0.0, "F4": 0.0}
    monkeypatch.setattr(module, "load_experiment", lambda: experiment)
    out, argv = splits_argv(tmp_path)
    module.main(argv)
    assert {i.family for i in load_draft(out / "dev.jsonl")} == {"F1", "F2"}


class RatedBackend(FakeBackend):
    """θ_0 emits a valid path at a per-draw rate; exemplar rollouts always assemble."""

    def __init__(self, rate):
        super().__init__()
        self.rate, self.rng = rate, random.Random(0)

    def generate(self, inputs, sampling, role="actor"):
        if inputs[0].prompt.startswith("Example"):
            return [Generation(f"Sure. {BASE} Done.") for _ in inputs]
        return [Generation(BASE if self.rng.random() < self.rate else "no tags") for _ in inputs]


@pytest.mark.parametrize("rate", [0.25, 0.5, 0.8, 0.95])
def test_plant_works_at_every_valid_rate(rate, tmp_path, xm, monkeypatch, capsys):
    monkeypatch.setattr("avrsi.models.load_backend", lambda model: RatedBackend(rate))
    drafts = tmp_path / "drafts"
    drafts.mkdir()
    for name in ("gate", "guard", "dev"):
        write_draft(xm[name], drafts / f"{name}.jsonl")
    sim = tmp_path / "sim.json"
    sim.write_text("{}")
    out = tmp_path / "planted"
    module = load("build_xm")
    monkeypatch.setattr(module, "label_disjoint_pairs", lambda *_: make_disjoint())  # the fixture's own tier
    module.main(["plant", "--model", "qwen3_omni", "--xm-dir", str(drafts), "--media-root", "m", "--similarity", str(sim),
                           "--out", str(out), "--draws", "1", "--split", "dev", "gate", "guard"])
    for name, n in (("gate", 500), ("guard", 256), ("dev", 256)):
        planted, report = load_planted(out / f"{name}_planted.json")
        assert len(planted) == n and report["missing"] == 0 and report["mode"] == "theta0" and report["prompt_digest"]
        assert report["first_round"] + report["redrawn"] + report["assembled"] >= n
        if name == "dev":
            assert all(p.item.family in ("F1", "F2") for p in planted)


def test_frame_sampling_reaches_the_openai_audit(monkeypatch, frozen, tmp_path):
    path, items = frozen
    seen = {}

    def fake_view(item, modalities, work_dir, runner=None, frames=None):
        seen[frames] = True
        return []

    module = load("audit")
    monkeypatch.setattr(module, "view_media", fake_view)
    monkeypatch.setattr(module, "make_judge", lambda config, scope: type("J", (), {"complete": lambda s, p, m: "A"})())
    a = tmp_path / "a.jsonl"
    write_preds(a, items[:2], lambda i: i.answer)
    module.main(["--items", str(path), "--media-root", "", "--ours", str(a), "--baseline", str(a), "--out", str(tmp_path / "o"),
                 "--judges", "openai", "--execute"])
    assert seen == {8: True}


def test_similarity_uses_the_configured_models(tmp_path, monkeypatch):
    seen = {}

    class Siglip:
        def __init__(self, model_id):
            seen["siglip"] = model_id

        def text_embeddings(self, classes):
            return {c: [1.0, float(i)] for i, c in enumerate(classes)}

    def clap(model_id, classes):
        seen["clap"] = model_id
        return {c: [1.0, 0.0] for c in classes}

    monkeypatch.setattr("avrsi.labels.SiglipScorer", Siglip)
    monkeypatch.setattr("avrsi.labels.clap_text_embeddings", clap)
    classes = tmp_path / "c.txt"
    classes.write_text("dog\ncat\n")
    module = load("build_xm")
    module.main(["similarity", "--classes", str(classes), "--out", str(tmp_path / "s.json")])
    table = module.similarity_table(tmp_path / "s.json")
    assert frozenset(("cat", "dog")) in table and seen["siglip"].startswith("google/") and seen["clap"].startswith("laion/")


def test_gate0_decide(tmp_path, capsys):
    report = {"detection": 0.9, "h2_gain": 0.2, "order_consistency": 0.9, "supported_on_control": 0.8, "parse_failure": {"a": 0.0}}
    for name in ("qwen3_omni", "avf"):
        (tmp_path / f"{name}.json").write_text(json.dumps(report))
    load("gate0").main(["decide", str(tmp_path / "qwen3_omni.json"), str(tmp_path / "avf.json")])
    assert json.loads(capsys.readouterr().out)["proceed"]




def test_smoke_job_asks_for_what_was_specified():
    text = (ROOT / "scripts" / "slurm" / "smoke_qwen3.sbatch").read_text()
    for line in ("--partition=gamma", "--account=gamma", "--qos=high", "--gres=gpu:rtxa6000:2", "--nodes=1", "-c 16", "--mem=120G", "--time=02:00:00"):
        assert f"#SBATCH {line}" in text
    assert "scripts/smoke_qwen3.py" in text and "run_lib" in text


@pytest.mark.parametrize("name", ["smoke_qwen3", "train_qwen3"])
def test_slurm_log_directory_is_in_the_tree(name):
    """Slurm opens --output before the job runs, so a fresh clone must already have the directory."""
    text = (ROOT / "scripts" / "slurm" / f"{name}.sbatch").read_text()
    output = next(line.split("=", 1)[1] for line in text.splitlines() if line.startswith("#SBATCH --output="))
    assert (ROOT / output).parent == ROOT / "outputs" / "slurm" and (ROOT / "outputs" / "slurm" / ".gitkeep").exists()
