import inspect

import pytest

from avrsi.models.base import AUDIO_KEYS, ModelInput, WithholdingError, assert_withheld, build_model_input
from avrsi.models.fake import FakeBackend
from avrsi.models.hf_common import HFBackend, messages, processor_kwargs
from avrsi.models.avf import AVFBackend
from avrsi.models.qwen import Qwen2_5OmniBackend, Qwen3OmniBackend
from avrsi.parse import Item

CLIP = Item("c", "q", "yes", video="clip.mp4")  # audio lives in the container
PAIRED = Item("c", "q", "yes", video="clip.mp4", audio="clip.wav")


class FakeProcessor:
    """Mimics the Omni processors: audio features come from `audio`, or from the video's own track
    when the flag is set, which is how a V-check could leak."""

    def __init__(self, flag="use_audio_in_video"):
        self.flag, self.last = flag, None

    def __call__(self, text, audio=None, videos=None, return_tensors=None, padding=None, **kwargs):
        self.last = dict(audio=audio, videos=videos, **kwargs)
        out = {"input_ids": [[1]], "attention_mask": [[1]]}
        if audio is not None or (videos is not None and kwargs.get(self.flag)):
            out |= {"input_features": [[0.0]], "feature_attention_mask": [[1]]}
        if videos is not None:
            out |= {"pixel_values_videos": [[0.0]], "video_grid_thw": [[1, 1, 1]]}
        return out


def encode(item, modalities, flag="use_audio_in_video", fill=None):
    x = build_model_input(item, modalities, "p", fill)
    processor = FakeProcessor(flag)
    encoded = processor(**processor_kwargs(x, "p", flag, audio_loader=lambda p: f"wave:{p}"))
    assert_withheld(x, encoded)
    return x, processor, encoded


@pytest.mark.parametrize("flag", ["use_audio_in_video", "load_audio_in_video"])
def test_v_check_cannot_leak_the_containers_audio(flag):
    x, processor, encoded = encode(CLIP, ("video",), flag)
    assert x.audio is None and processor.last["audio"] is None and processor.last[flag] is False
    assert not set(encoded) & set(AUDIO_KEYS)


@pytest.mark.parametrize("flag", ["use_audio_in_video", "load_audio_in_video"])
def test_a_check_gets_no_frames(flag):
    x, processor, encoded = encode(CLIP, ("audio",), flag)
    assert x.video is None and processor.last["videos"] is None and processor.last[flag] is False
    assert processor.last["audio"] == ["wave:clip.mp4"]  # taken from the container
    assert "pixel_values_videos" not in encoded


def test_joint_views_send_both_with_flag():
    _, processor, encoded = encode(CLIP, ("audio", "video"))
    assert processor.last["use_audio_in_video"] is True and processor.last["audio"] and processor.last["videos"]
    assert "input_features" in encoded and "pixel_values_videos" in encoded


def test_separate_audio_file_is_used():
    assert build_model_input(PAIRED, ("audio",), "p").audio == "clip.wav"


def test_assert_catches_leak_if_flag_were_set():
    x = build_model_input(CLIP, ("video",), "p")
    kwargs = processor_kwargs(x, "p", "use_audio_in_video")
    kwargs["use_audio_in_video"] = True  # the bug this guards against
    with pytest.raises(WithholdingError, match="audio withheld"):
        assert_withheld(x, FakeProcessor()(**kwargs))


def test_assert_catches_missing_requested_modality():
    x = build_model_input(CLIP, ("audio", "video"), "p")
    with pytest.raises(WithholdingError, match="absent"):
        assert_withheld(x, {"input_ids": [[1]], "pixel_values_videos": [[0.0]]})


def test_withheld_really_changes_inputs():
    keys = {m: set(encode(CLIP, m)[2]) for m in (("audio",), ("video",), ("audio", "video"))}
    assert keys[("audio",)] != keys[("video",)] != keys[("audio", "video")] and keys[("audio",)] != keys[("audio", "video")]


def test_fill_carries_no_signal_and_no_source():
    x = build_model_input(CLIP, ("video",), "p", fill="silent_track")
    assert x.audio is None and x.fill == "silent_track"
    kwargs = processor_kwargs(x, "p", "use_audio_in_video")
    assert (kwargs["audio"][0] == 0).all() and kwargs["use_audio_in_video"] is False
    assert_withheld(x, FakeProcessor()(**kwargs))


def test_black_frames_fill():
    x = build_model_input(CLIP, ("audio",), "p", fill="black_frames")
    kwargs = processor_kwargs(x, "p", "use_audio_in_video", audio_loader=lambda p: p)
    assert (kwargs["videos"][0] == 0).all()


def test_fill_not_applied_to_joint_view():
    assert build_model_input(CLIP, ("audio", "video"), "p", fill="silent_track").fill is None


def test_wrong_kind_of_fill_ignored():
    assert build_model_input(CLIP, ("audio",), "p", fill="silent_track").fill is None


def test_unimodal_sample_stays_unimodal():
    x = build_model_input(Item("c", "q", "yes", video="a.mp4", modalities=("video",)), ("audio", "video"), "p")
    assert x.audio is None and x.video == "a.mp4"


def test_missing_media_raises():
    with pytest.raises(WithholdingError):
        build_model_input(Item("c", "q", "yes"), ("audio", "video"), "p")


def test_unknown_modality_raises():
    with pytest.raises(ValueError):
        build_model_input(CLIP, ("smell",), "p")


def test_messages_have_placeholder_only_for_sent_media():
    types = lambda x: [c["type"] for c in messages(x)[0]["content"]]  # noqa: E731
    assert types(ModelInput("p", audio="a")) == ["audio", "text"]
    assert types(ModelInput("p", video="v")) == ["video", "text"]
    assert types(ModelInput("p", video="v", audio="a")) == ["audio", "video", "text"]
    assert types(ModelInput("p", video="v", fill="silent_track")) == ["audio", "video", "text"]


@pytest.mark.parametrize("cls", [Qwen3OmniBackend, Qwen2_5OmniBackend, AVFBackend])
def test_real_wrappers_are_marked_untested(cls):
    assert issubclass(cls, HFBackend) and "UNTESTED" in inspect.getmodule(cls).__doc__


def test_hf_encode_applies_the_assert():
    source = inspect.getsource(HFBackend._encode)
    assert "assert_withheld" in source and "processor_kwargs" in source


def test_lora_refuses_unresolved_targets():
    backend = HFBackend.__new__(HFBackend)
    backend.config = {"lora_target_modules": None}
    with pytest.raises(ValueError, match="VERIFY"):
        backend.attach_lora(32)


def test_fake_critic_perceives_only_given_modalities():
    backend = FakeBackend()
    prompt = "You are checking claims\n1. The video shows dog.\n"
    assert backend._check(ModelInput(prompt, audio="fake://dog"), "critic") == "1: unverifiable"
    assert backend._check(ModelInput(prompt, video="fake://dog"), "critic") == "1: supported"
    assert backend._check(ModelInput(prompt, video="fake://piano"), "critic") == "1: unsupported"


def test_fake_gradients_flow_only_through_actor():
    backend = FakeBackend()
    from avrsi.models.base import Sampling
    gen = backend.generate([ModelInput("p")], Sampling(1.0, 1.0, True, 32))[0]
    assert backend.score_logprobs([None], [gen])[0].requires_grad
    assert not backend.score_logprobs([None], [gen], "reference")[0].requires_grad
