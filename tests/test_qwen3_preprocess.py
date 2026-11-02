"""The real Qwen3OmniMoeProcessor from the local snapshot on a generated MP4, then through the tiny real
thinker: AV, A-only and V-only with withholding asserted on the tensors."""
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch

pytest.importorskip("peft")
pytest.importorskip("av")
from qwen3_tiny import BASE, tiny_thinker  # noqa: E402
from transformers import Qwen3OmniMoeProcessor  # noqa: E402

from avrsi import ffmpeg  # noqa: E402
from avrsi.critic import A_CHECK, AV_PAIR, V_CHECK, VIEW_MODALITIES  # noqa: E402
from avrsi.models.base import ACTOR, CRITIC, REFERENCE, Generation, ModelInput, Sampling, WithholdingError, build_model_input  # noqa: E402
from avrsi.models.hf_common import load_audio, media_kwargs, messages, processor_kwargs, read_video, sampled_video_kwargs  # noqa: E402
from avrsi.models.qwen import Qwen3OmniBackend  # noqa: E402
from avrsi.parse import Item  # noqa: E402

SNAPSHOT = Path("/fs/gamma-projects/audio/.cache/hub/models--Qwen--Qwen3-Omni-30B-A3B-Captioner/snapshots/a2bd106cbf527db5676e79662674da22b0545ec0")
pytestmark = pytest.mark.skipif(not (SNAPSHOT / "vocab.json").exists(), reason="snapshot with tokenizer files absent")
AUDIO, VIDEO = 151675, 151656
SECONDS, FPS, FRAMES = 4, 10, 8
FLAG = "use_audio_in_video"


def make_clip(path, audio=True):
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", f"testsrc2=size=320x240:rate={FPS}:duration={SECONDS}"]
    if audio:
        cmd += ["-f", "lavfi", "-i", f"sine=frequency=440:duration={SECONDS}", "-c:a", "aac", "-shortest"]
    ffmpeg.run([*cmd, "-c:v", "libx264", "-pix_fmt", "yuv420p", str(path)], check=True)
    return str(path)


@pytest.fixture(scope="module")
def clip(tmp_path_factory):
    return make_clip(tmp_path_factory.mktemp("clip") / "c.mp4")


@pytest.fixture(scope="module")
def processor():
    return Qwen3OmniMoeProcessor.from_pretrained(SNAPSHOT)


class Tiny(Qwen3OmniBackend):
    def __init__(self, config, processor):
        self._processor = processor
        super().__init__({**BASE, "num_frames": FRAMES, "lora_exclude_modules": r"(audio_tower|visual)\..*", **config}, device="cpu")

    def _load(self):
        return self._processor, tiny_thinker()


@pytest.fixture(scope="module")
def backend(processor):
    built = Tiny({}, processor)
    torch.manual_seed(0)
    built.attach_lora(4)
    for param in built.trainable_parameters():
        param.data.add_(0.05)
    return built


def views(clip):
    item = Item("c", "q", "yes", video=clip)
    return {name: build_model_input(item, modalities, "Is there a tone?") for name, modalities in VIEW_MODALITIES.items()}


def encode(backend, inputs):
    encoded, flag = backend._encode(inputs)
    return encoded, flag, encoded["input_ids"][0]


def test_ffmpeg_helper_finds_a_binary(clip):
    assert Path(ffmpeg.ffmpeg_exe()).exists() and Path(clip).stat().st_size > 0


def test_ffprobe_falls_back_to_pyav(clip):
    out = ffmpeg.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", clip], check=True, capture_output=True, text=True)
    assert float(out.stdout) == pytest.approx(SECONDS, abs=0.2)


def test_mux_commands_run_through_the_helper(clip, tmp_path):
    from avrsi.perturb import mux_command, strip_audio_command

    ffmpeg.run(mux_command(clip, None, str(tmp_path / "silent.mp4")), check=True)
    assert np.abs(load_audio(str(tmp_path / "silent.mp4"))).max() == 0
    ffmpeg.run(strip_audio_command(clip, str(tmp_path / "v.mp4")), check=True)
    with pytest.raises(WithholdingError):
        load_audio(str(tmp_path / "v.mp4"))


def test_decoders(clip):
    frames, metadata = read_video(clip)
    assert frames.shape == (SECONDS * FPS, 240, 320, 3) and frames.dtype == np.uint8 and metadata.total_num_frames == SECONDS * FPS
    wave = load_audio(clip)
    assert wave.dtype == np.float32 and len(wave) == pytest.approx(SECONDS * 16000, rel=0.08) and np.abs(wave).max() > 0.05


def test_frame_count_reaches_the_processor_with_a_consistent_rate(backend, clip):
    for n in (FRAMES, 20):
        settings = sampled_video_kwargs(n, read_video(clip)[1])
        assert settings["do_sample_frames"] is True and settings["fps"] == pytest.approx(n / SECONDS, rel=1e-3)
    encoded, _, _ = encode(backend, views(clip)[V_CHECK])
    assert encoded["video_grid_thw"].tolist() == [[FRAMES // 2, 20, 28]]  # 320x240 is upscaled to the processor's 128 tokens, at 16 px
    seconds_per_grid = float(encoded["video_second_per_grid"][0])
    assert seconds_per_grid == pytest.approx(2 * SECONDS / FRAMES, rel=1e-3)  # not 2 / 1.0, the processor's default
    assert encoded["video_second_per_grid"].dtype != torch.bfloat16


def test_short_clip_samples_what_it_has(backend, tmp_path):
    path = str(tmp_path / "short.mp4")
    ffmpeg.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", f"testsrc2=size=320x240:rate={FPS}:duration=0.6", "-f", "lavfi", "-i",
                "sine=duration=0.6", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", path], check=True)
    encoded, _, _ = encode(backend, ModelInput("q", video=path, audio=path))
    assert encoded["video_grid_thw"][0, 0] == 3


def test_num_frames_alone_would_be_ignored(processor, clip):
    """The appendix note: without do_sample_frames every frame goes through."""
    frames = read_video(clip)[0]
    text = processor.apply_chat_template(messages(ModelInput("q", video=clip)), add_generation_prompt=True, tokenize=False)
    out = processor(text=[text], videos=[frames], num_frames=FRAMES, return_tensors="pt")
    assert out["video_grid_thw"][0, 0] == SECONDS * FPS // 2


def test_av_view_carries_both_streams(backend, clip):
    inputs = views(clip)[AV_PAIR]
    encoded, flag, ids = encode(backend, inputs)
    assert flag is True and inputs.modalities == ("audio", "video")
    assert encoded["input_features"].dtype == torch.bfloat16 and encoded["pixel_values_videos"].dtype == torch.bfloat16
    n_video = int(encoded["video_grid_thw"].prod()) // 4
    assert (ids == VIDEO).sum() == n_video and (ids == AUDIO).sum() > 0
    audio_at, video_at = (ids == AUDIO).nonzero(), (ids == VIDEO).nonzero()
    assert audio_at.min() < video_at.max() and video_at.min() < audio_at.max()  # interleaved


def test_a_only_view_withholds_the_frames(backend, clip):
    inputs = views(clip)[A_CHECK]
    encoded, flag, ids = encode(backend, inputs)
    assert flag is False and inputs.video is None
    assert not {"pixel_values_videos", "video_grid_thw", "video_second_per_grid"} & set(encoded.keys())
    assert (ids == VIDEO).sum() == 0 and (ids == AUDIO).sum() > 0


def test_v_only_view_withholds_the_audio_of_the_same_file(backend, clip):
    inputs = views(clip)[V_CHECK]
    encoded, flag, ids = encode(backend, inputs)
    assert flag is False and inputs.audio is None
    assert not {"input_features", "feature_attention_mask"} & set(encoded.keys())
    assert (ids == AUDIO).sum() == 0 and (ids == VIDEO).sum() > 0


def test_withheld_streams_leave_the_other_unchanged(backend, clip):
    all_views = views(clip)
    av = encode(backend, all_views[AV_PAIR])[0]
    assert torch.equal(av["pixel_values_videos"], encode(backend, all_views[V_CHECK])[0]["pixel_values_videos"])
    assert torch.equal(av["input_features"], encode(backend, all_views[A_CHECK])[0]["input_features"])


def test_a_flag_leak_would_be_caught(clip, processor):
    """Audio placeholder consumed by a video under the flag with no audio given: the processor refuses."""
    inputs = ModelInput("q", video=clip)
    text = processor.apply_chat_template(messages(inputs), add_generation_prompt=True, tokenize=False)
    frames = read_video(clip)[0]
    with pytest.raises(StopIteration):
        processor(text=[text], videos=[frames], use_audio_in_video=True, return_tensors="pt")


@pytest.mark.parametrize("view", [AV_PAIR, A_CHECK, V_CHECK])
@pytest.mark.parametrize("role", [ACTOR, CRITIC, REFERENCE])
def test_real_thinker_generates_in_each_view(backend, clip, view, role):
    gen = backend.generate([views(clip)[view]], Sampling(0.0, 1.0, False, 3), role)[0]
    assert len(gen.token_ids) == 3


@pytest.mark.parametrize("view", [AV_PAIR, A_CHECK, V_CHECK])
def test_real_thinker_scores_and_backprops_in_each_view(backend, clip, view):
    backend.model.zero_grad()
    logps = backend.score_logprobs([views(clip)[view]], [Generation("", [11, 12, 13])], ACTOR)[0]
    assert logps.shape == (3,) and logps.dtype == torch.float32
    logps.sum().backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in backend.trainable_parameters())


def test_withholding_changes_the_output(backend, clip):
    all_views = views(clip)
    gen = Generation("", [11, 12, 13])
    scores = [backend.score_logprobs([all_views[v]], [gen], REFERENCE)[0] for v in (AV_PAIR, A_CHECK, V_CHECK)]
    assert not torch.allclose(scores[0], scores[1]) and not torch.allclose(scores[0], scores[2])


def test_multimodal_batch_equals_single(processor, clip):
    single, batched = Tiny({}, processor), Tiny({"logprob_batch": 2}, processor)
    for b in (single, batched):
        torch.manual_seed(0)
        b.attach_lora(4)
        for param in b.trainable_parameters():
            param.data.add_(0.05)
    inputs = [views(clip)[AV_PAIR]] * 2
    gens = [Generation("", [11, 12, 13]), Generation("", [14, 15])]
    for a, b in zip(single.score_logprobs(inputs, gens, ACTOR), batched.score_logprobs(inputs, gens, ACTOR)):
        assert torch.allclose(a, b, atol=1e-4)


def test_k_rollouts_on_a_multimodal_prompt(processor, clip):
    backend = Tiny({"generate_batch": 4}, processor)
    torch.manual_seed(0)
    backend.attach_lora(4)
    gens = backend.generate([views(clip)[AV_PAIR]] * 4, Sampling(1.0, 1.0, True, 4))
    assert [len(g.token_ids) for g in gens] == [4] * 4 and len({tuple(g.token_ids) for g in gens}) > 1


def test_gradient_checkpointing_with_media(processor, clip):
    plain, checkpointed = Tiny({}, processor), Tiny({"gradient_checkpointing": True}, processor)
    gen = Generation("", [11, 12, 13])
    grads = []
    for b in (plain, checkpointed):
        torch.manual_seed(0)
        b.attach_lora(4)
        for param in b.trainable_parameters():
            param.data.add_(0.05)
        b.score_logprobs([views(clip)[AV_PAIR]], [gen], ACTOR)[0].sum().backward()
        grads.append([p.grad for p in b.trainable_parameters()])
    for a, b in zip(*grads):
        assert torch.allclose(a, b, atol=1e-5)


def test_media_settings_for_the_real_config():
    from avrsi.config import load_model

    config = load_model("qwen3_omni")
    assert media_kwargs(config) == {"num_frames": 20} and config["num_frames"] % 2 == 0
    assert processor_kwargs(ModelInput("q", video="v.mp4"), "t", FLAG, video_loader=lambda p: p, settings={})[FLAG] is False


def test_subprocess_is_not_needed_for_decoding(clip, monkeypatch):
    read_video.cache_clear(), load_audio.cache_clear()
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: pytest.fail("decoding shelled out"))
    assert len(read_video(clip)[0]) and len(load_audio(clip))
