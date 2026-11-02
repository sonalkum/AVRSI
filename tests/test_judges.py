import json
import logging
import random
from types import SimpleNamespace

import httpx
import pytest

from avrsi.config import load_experiment
from avrsi.judges import GeminiJudge, JudgeError, Media, OpenAIJudge, SpendCapExceeded, load_key, make_judge
from avrsi.judges.audit import audit_pair
from avrsi.judges.base import Secret, SpendCap
from avrsi.judges.media import sample_frames, view_media
from avrsi.critic import JudgeExecutor, CriticPass, A_CHECK, parse_check, parse_pair
from avrsi.parse import Item

KEY = "sk-test-CANARY-0123456789"


def gemini_ok(text="1: supported"):
    return {"candidates": [{"content": {"parts": [{"text": text}]}}]}


def openai_ok(text="A"):
    return {"choices": [{"message": {"content": text}}]}


class Recorder:
    def __init__(self, reply, status=200):
        self.requests, self.reply, self.status = [], reply, status

    def __call__(self, request):
        self.requests.append(request)
        status = self.status.pop(0) if isinstance(self.status, list) else self.status
        if status != 200:
            return httpx.Response(status, json={"error": f"bad key {KEY}"})
        return httpx.Response(200, json=self.reply)


def config(name, **over):
    return {**load_experiment()["judges"][name], "key_env": "AVRSI_TEST_KEY", "key_file": None, **over}


@pytest.fixture(autouse=True)
def key(monkeypatch):
    monkeypatch.setenv("AVRSI_TEST_KEY", KEY)


def gemini(tmp_path, recorder, **over):
    return GeminiJudge(config("gemini", **over), transport=httpx.MockTransport(recorder), cache_dir=tmp_path, sleep=lambda s: None)


def openai(tmp_path, recorder, **over):
    return OpenAIJudge(config("openai", model="test-model", **over), transport=httpx.MockTransport(recorder), cache_dir=tmp_path, sleep=lambda s: None)


def test_gemini_key_only_in_header(tmp_path):
    rec = Recorder(gemini_ok())
    assert gemini(tmp_path, rec).complete("hello") == "1: supported"
    request = rec.requests[0]
    assert request.headers["x-goog-api-key"] == KEY
    assert KEY not in str(request.url) and KEY.encode() not in request.content
    assert "generativelanguage.googleapis.com" in request.url.host and ":generateContent" in request.url.path


def test_openai_key_only_in_header(tmp_path):
    rec = Recorder(openai_ok())
    assert openai(tmp_path, rec).complete("hello") == "A"
    request = rec.requests[0]
    assert request.headers["authorization"] == f"Bearer {KEY}"
    assert KEY not in str(request.url) and KEY.encode() not in request.content


@pytest.mark.parametrize("make,ok", [(gemini, gemini_ok()), (openai, openai_ok())])
def test_key_never_logged_at_debug(tmp_path, caplog, make, ok):
    caplog.set_level(logging.DEBUG)
    make(tmp_path, Recorder(ok)).complete("hello")
    with pytest.raises(JudgeError) as error:
        make(tmp_path / "x", Recorder(ok, status=401)).complete("hello")
    assert KEY not in caplog.text and KEY not in str(error.value) and "401" in str(error.value)


def test_key_not_in_repr_or_disk(tmp_path):
    judge = gemini(tmp_path, Recorder(gemini_ok()))
    judge.complete("hello")
    assert KEY not in repr(judge) and KEY not in repr(judge._key) and KEY not in f"{judge._key}"
    assert all(KEY not in p.read_text() for p in tmp_path.rglob("*.json"))


def test_transport_error_message_is_clean(tmp_path):
    def boom(request):
        raise httpx.ConnectError(f"failed with {request.headers['x-goog-api-key']}")

    with pytest.raises(JudgeError) as error:
        gemini(tmp_path, boom).complete("hello")
    assert KEY not in str(error.value) and error.value.__cause__ is None and error.value.__suppress_context__


def test_key_from_file(tmp_path, monkeypatch):
    monkeypatch.delenv("AVRSI_TEST_KEY")
    path = tmp_path / "k"
    path.write_text(f"{KEY}\n")
    assert load_key("AVRSI_TEST_KEY", str(path)).reveal() == KEY


def test_env_wins_over_file(tmp_path, monkeypatch):
    monkeypatch.setenv("AVRSI_TEST_KEY", "env-key")
    path = tmp_path / "k"
    path.write_text(KEY)
    assert load_key("AVRSI_TEST_KEY", str(path)).reveal() == "env-key"


def test_missing_key_message_has_no_value(monkeypatch):
    monkeypatch.delenv("AVRSI_TEST_KEY")
    with pytest.raises(JudgeError, match="AVRSI_TEST_KEY"):
        load_key("AVRSI_TEST_KEY", None)


def test_secret_is_masked():
    assert KEY not in repr(Secret(KEY)) and KEY not in str(Secret(KEY))


def test_cache_makes_rerun_free(tmp_path):
    rec = Recorder(gemini_ok())
    first = gemini(tmp_path, rec)
    assert first.complete("hello") == first.complete("hello")
    second = gemini(tmp_path, rec)
    assert second.complete("hello") == "1: supported"
    assert len(rec.requests) == 1 and second.cap.calls == 1


def test_cache_key_covers_content_and_model(tmp_path):
    rec = Recorder(gemini_ok())
    judge = gemini(tmp_path, rec)
    judge.complete("one")
    judge.complete("two")
    judge.complete("one", [Media("audio", "audio/wav", data=b"x")])
    judge.complete("one", [Media("audio", "audio/wav", data=b"y")])
    gemini(tmp_path, rec, model="other").complete("one")
    assert len(rec.requests) == 5


def test_cache_follows_file_content_not_path(tmp_path):
    rec = Recorder(gemini_ok())
    judge = gemini(tmp_path, rec)
    media = tmp_path / "a.wav"
    media.write_bytes(b"one")
    judge.complete("p", [Media("audio", "audio/wav", path=str(media))])
    media.write_bytes(b"two")
    judge.complete("p", [Media("audio", "audio/wav", path=str(media))])
    assert len(rec.requests) == 2


def test_failures_are_not_cached(tmp_path):
    rec = Recorder(gemini_ok(), status=[400, 200])
    judge = gemini(tmp_path, rec)
    with pytest.raises(JudgeError):
        judge.complete("hello")
    assert judge.complete("hello") == "1: supported"


def test_call_cap_blocks_before_transport(tmp_path):
    rec = Recorder(gemini_ok())
    judge = gemini(tmp_path, rec, max_calls=2)
    judge.complete("a"), judge.complete("b")
    with pytest.raises(SpendCapExceeded):
        judge.complete("c")
    assert len(rec.requests) == 2
    assert judge.complete("a") == "1: supported"  # cached calls stay free


def test_usd_cap(tmp_path):
    rec = Recorder(gemini_ok())
    judge = gemini(tmp_path, rec, usd_per_call=0.4, max_usd=1.0)
    judge.complete("a"), judge.complete("b")
    with pytest.raises(SpendCapExceeded):
        judge.complete("c")


def test_cap_persists_across_runs(tmp_path):
    cap = SpendCap(tmp_path / "ledger.json", 1, 1.0, None)
    cap.charge()
    with pytest.raises(SpendCapExceeded):
        SpendCap(tmp_path / "ledger.json", 1, 1.0, None).charge()


def test_retry_on_429_charges_each_attempt(tmp_path):
    rec = Recorder(gemini_ok(), status=[429, 503, 200])
    judge = gemini(tmp_path, rec)
    assert judge.complete("a") == "1: supported"
    assert len(rec.requests) == 3 and judge.cap.calls == 3


def test_retry_gives_up(tmp_path):
    with pytest.raises(JudgeError):
        gemini(tmp_path, Recorder(gemini_ok(), status=[503] * 5)).complete("a")


def test_gemini_media_inline(tmp_path):
    rec = Recorder(gemini_ok())
    gemini(tmp_path, rec).complete("p", [Media("video", "video/mp4", data=b"abc")])
    part = json.loads(rec.requests[0].content)["contents"][0]["parts"][0]
    assert part["inline_data"]["mime_type"] == "video/mp4"


def test_openai_frames_plus_audio_no_video(tmp_path):
    rec = Recorder(openai_ok())
    judge = openai(tmp_path, rec)
    judge.complete("p", [Media("image", "image/jpeg", data=b"a"), Media("audio", "audio/wav", data=b"b")])
    kinds = [c["type"] for c in json.loads(rec.requests[0].content)["messages"][0]["content"]]
    assert kinds == ["text", "image_url", "input_audio"]
    with pytest.raises(JudgeError):
        judge.complete("p", [Media("video", "video/mp4", data=b"c")])


def test_unset_openai_model_refuses_before_any_call(tmp_path):
    rec = Recorder(openai_ok())
    judge = OpenAIJudge(config("openai"), transport=httpx.MockTransport(rec), cache_dir=tmp_path)
    with pytest.raises(JudgeError, match="VERIFY"):
        judge.complete("p")
    assert not rec.requests


def test_model_ids_come_from_config():
    judges = load_experiment()["judges"]
    assert judges["gemini"]["model"] == "gemini-3.8-flash"
    assert "max_calls" in judges["gemini"] and "max_usd" in judges["openai"]


def test_config_has_no_secret_values():
    judges = load_experiment()["judges"]
    assert all(not k.endswith("api_key") and "key" not in str(v["model"]) for k, v in judges.items())


def test_make_judge(tmp_path):
    judge = make_judge(config("gemini"), transport=httpx.MockTransport(Recorder(gemini_ok())), cache_dir=tmp_path)
    assert isinstance(judge, GeminiJudge)


def test_view_media_never_sends_audio_track_to_video_view(tmp_path):
    calls = []
    item = Item("c", "q", "yes", video="v.mp4")
    media = view_media(item, ("video",), tmp_path, runner=lambda cmd, check: calls.append(cmd))
    assert "-an" in calls[0] and media[0].path.endswith("noaudio.mp4")


def test_view_media_audio_only_extracts(tmp_path):
    calls = []
    media = view_media(Item("c", "q", "yes", video="v.mp4"), ("audio",), tmp_path, runner=lambda cmd, check: calls.append(cmd))
    assert "-vn" in calls[0] and media[0].kind == "audio"
    assert view_media(Item("c", "q", "yes", video="v.mp4", audio="a.wav"), ("audio",), tmp_path)[0].path == "a.wav"


def test_view_media_both_sends_container(tmp_path):
    assert view_media(Item("c", "q", "yes", video="v.mp4"), ("audio", "video"), tmp_path)[0].kind == "video"


def test_judge_executor_passes_view_modalities():
    seen = []

    class J:
        def complete(self, prompt, media):
            seen.append(media)
            return "1: supported"

    executor = JudgeExecutor(J(), lambda item, modalities: modalities)
    executor.run(Item("c", "q", "yes"), [CriticPass(A_CHECK, "p")])
    assert seen == [("audio",)] and executor.calls == 1


@pytest.mark.parametrize("reply,expected", [("Tie", 0.5), ("junk", None)])
def test_audit_pair_tie_and_failure(reply, expected):
    class J:
        def complete(self, prompt, media):
            return reply

    assert audit_pair(J(), Item("c", "q", "yes"), "ours", "base", [], random.Random(1)) == expected


def test_audit_is_blind_and_maps_order_back():
    class J:
        def complete(self, prompt, media):
            return "A" if prompt.index("OURS") < prompt.index("BASE") else "B"  # always picks ours

    rng = random.Random(0)
    assert all(audit_pair(J(), Item("c", "q", "yes"), "OURS", "BASE", [], rng) == 1.0 for _ in range(20))


def test_audit_order_is_randomised():
    orders = []

    class J:
        def complete(self, prompt, media):
            orders.append(prompt.index("OURS") < prompt.index("BASE"))
            return "Tie"

    rng = random.Random(0)
    for _ in range(40):
        audit_pair(J(), Item("c", "q", "yes"), "OURS", "BASE", [], rng)
    assert 5 < sum(orders) < 35


def flaky(failures, reply):
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        if calls["n"] <= failures:
            raise httpx.ConnectTimeout("slow")
        return httpx.Response(200, json=reply)

    return handler, calls


def test_transport_errors_retry_with_backoff(tmp_path):
    handler, calls = flaky(2, gemini_ok("ok"))
    slept = []
    judge = GeminiJudge(config("gemini"), transport=httpx.MockTransport(handler), cache_dir=tmp_path, sleep=slept.append)
    assert judge.complete("hello") == "ok"
    assert calls["n"] == 3 and slept == [1.0, 2.0] and judge.cap.calls == 3


def test_transport_errors_give_up_after_max_attempts(tmp_path):
    handler, calls = flaky(9, gemini_ok())
    with pytest.raises(JudgeError, match="transport error ConnectTimeout"):
        gemini(tmp_path, handler).complete("hello")
    assert calls["n"] == 3


@pytest.mark.parametrize("reply", [{"candidates": []}, {}, {"candidates": [{"finishReason": "SAFETY"}]}])
def test_gemini_200_without_text_is_empty_and_cached(tmp_path, reply):
    rec = Recorder(reply)
    judge = gemini(tmp_path, rec)
    assert judge.complete("hello") == "" and judge.complete("hello") == ""
    assert len(rec.requests) == 1


def test_openai_200_null_content_is_empty(tmp_path):
    assert openai(tmp_path, Recorder({"choices": [{"message": {"content": None}}]})).complete("hello") == ""


def test_empty_reply_is_a_parse_failure_not_a_crash():
    assert parse_check("", 2) is None and parse_pair("") is None


def test_caps_are_per_scope_with_own_ledger(tmp_path):
    rec = Recorder(gemini_ok())
    cfg = {**config("gemini"), "caps": {"A3": {"max_calls": 1, "max_usd": None}, "A4": {"max_calls": 2, "max_usd": None}}}
    a3 = make_judge(cfg, "A3", transport=httpx.MockTransport(rec), cache_dir=tmp_path, sleep=lambda s: None)
    a4 = make_judge(cfg, "A4", transport=httpx.MockTransport(rec), cache_dir=tmp_path, sleep=lambda s: None)
    a3.complete("one")
    with pytest.raises(SpendCapExceeded):
        a3.complete("two")
    a4.complete("two"), a4.complete("three")
    assert (tmp_path / "gemini" / "ledger_A3.json").exists() and (tmp_path / "gemini" / "ledger_A4.json").exists()


def test_a3_a4_caps_cover_their_volume():
    caps = load_experiment()["judges"]["gemini"]["caps"]
    assert caps["A3"]["max_calls"] >= 716_800 and caps["A4"]["max_calls"] >= 716_800


def test_usd_per_call_needs_a_usd_cap(tmp_path):
    with pytest.raises(JudgeError, match="max_usd"):
        SpendCap(tmp_path / "l.json", 10, None, 0.01)


def media_runner(calls):
    def run(cmd, check, **kwargs):
        calls.append(cmd)
        return SimpleNamespace(stdout="10.0\n")
    return run


def test_sample_frames_uniform_times(tmp_path):
    calls = []
    frames = sample_frames("v.mp4", 4, tmp_path, "c", runner=media_runner(calls))
    assert calls[0][0] == "ffprobe" and [c[c.index("-ss") + 1] for c in calls[1:]] == ["1.250", "3.750", "6.250", "8.750"]
    assert all(c[0] == "ffmpeg" and "v.mp4" in c for c in calls[1:])
    assert [m.kind for m in frames] == ["image"] * 4 and frames[0].mime == "image/jpeg"


def test_view_media_frames_with_audio_and_no_video_part(tmp_path):
    calls = []
    item = Item("c", "q", "yes", video="v.mp4")
    media = view_media(item, ("audio", "video"), tmp_path, runner=media_runner(calls), frames=2)
    assert [m.kind for m in media] == ["image", "image", "audio"] and "-vn" in calls[-1]


def test_view_media_frames_never_add_audio_to_a_video_view(tmp_path):
    media = view_media(Item("c", "q", "yes", video="v.mp4"), ("video",), tmp_path, runner=media_runner([]), frames=2)
    assert {m.kind for m in media} == {"image"}


def test_openai_accepts_the_frame_view(tmp_path):
    rec = Recorder(openai_ok())
    (tmp_path / "x.jpg").write_bytes(b"a")
    openai(tmp_path, rec).complete("p", [Media("image", "image/jpeg", str(tmp_path / "x.jpg"))])
    assert json.loads(rec.requests[0].content)["messages"][0]["content"][1]["type"] == "image_url"
