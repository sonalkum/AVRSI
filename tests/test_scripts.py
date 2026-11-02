import importlib.util
import json
from pathlib import Path

import pytest

from avrsi.perturb import freeze

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


def test_evaluate_compare(frozen, tmp_path, capsys):
    path, items = frozen
    a, b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    write_preds(a, items, lambda i: i.answer)
    write_preds(b, items, lambda i: "no" if not i.options else i.options[0])
    load("evaluate").main(["compare", "--a", f"xm={a}", "--b", f"xm={b}", "--annotation", f"xm={path}", "--media-root", "xm=", "--bootstraps", "30"])
    out = json.loads(capsys.readouterr().out)
    assert out["delta"] > 0.3 and out["lower"] > 0


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
    rows = ["clip_id,source_video,label,siglip_score,partner_scores"]
    classes = ["dog", "piano", "rain", "siren"]
    rows += [f"c{i},s{i},{classes[i % 4]},0.9,{{}}" for i in range(400)]
    csv = tmp_path / "clips.csv"
    csv.write_text("\n".join(rows))
    sim = tmp_path / "sim.json"
    sim.write_text(json.dumps({f"{a}|{b}": 0.1 for i, a in enumerate(classes) for b in classes[i + 1:]}))
    out = tmp_path / "xm"
    load("build_xm").main(["splits", "--clips", str(csv), "--similarity", str(sim), "--threshold", "0.5", "--out", str(out)])
    assert (out / "test.jsonl.sha256").exists() and len((out / "gate.jsonl").read_text().splitlines()) == 500


def test_gate0_decide(tmp_path, capsys):
    report = {"detection": 0.9, "h2_gain": 0.2, "order_consistency": 0.9, "supported_on_control": 0.8, "parse_failure": {"a": 0.0}}
    for name in ("qwen3_omni", "avf"):
        (tmp_path / f"{name}.json").write_text(json.dumps(report))
    load("gate0").main(["decide", str(tmp_path / "qwen3_omni.json"), str(tmp_path / "avf.json")])
    assert json.loads(capsys.readouterr().out)["proceed"]


