#!/usr/bin/env python
"""Two real A8 steps on real weights, to measure step time and memory before any pilot. Not science.

The default weights are the local Captioner snapshot: same thinker class and sizes, but tuned for audio
captions with no prompt, so it ignores the format instruction. It is for local checks only. Paths are
therefore forced into the format (--force-format) or no critic pass would ever run. Use the Instruct
weights for anything that is read as a result.

The local VGGSound tree holds WAVs with no video, so each clip gets a generated test pattern: the audio is
real, the frames are not, and F2-style answers mean nothing here. The guard set is a tiny XM-Guard stand-in.

    torchrun --nproc_per_node=<GPUs / gpus-per-rank> scripts/smoke_qwen3.py --gpus-per-rank 2 --out <dir>"""
import argparse
import json
import os
import random
import time
from dataclasses import replace
from pathlib import Path

import torch

from avrsi import dist, ffmpeg
from avrsi.config import load_experiment, load_model, resolve_arm
from avrsi.models.base import ACTOR, Generation, Sampling, WithholdingError, build_model_input
from avrsi.models.qwen import Qwen3OmniBackend
from avrsi.parse import Item
from avrsi.perturb import F1_T, XMItem, build_planted
from avrsi.sync import GuardSet, modality_probes
from avrsi.train import Samplings, train

HUB = "/fs/gamma-projects/audio/.cache/hub"
CAPTIONER = f"{HUB}/models--Qwen--Qwen3-Omni-30B-A3B-Captioner/snapshots/a2bd106cbf527db5676e79662674da22b0545ec0"
VGGSOUND = "/fs/nexus-projects/brain_project/naacl_audio/VGGSound/scratch/shared/beegfs/hchen/train_data/VGGSound_final/video"
LABELS = "/fs/nexus-projects/brain_project/naacl_audio/vgg_match.json"
PATTERN = "a test pattern"
BASE = ("<audio>A sound runs through the clip. The level changes over time. Some noise sits behind it.</audio>"
        "<visual>A pattern fills the frame. The colours move slowly. Nothing else appears.</visual>"
        "<fusion>The sound and picture are unrelated.</fusion><answer>{}</answer>")


class ForcedFormat:
    """Delegates to a backend; an actor rollout that breaks the format is replaced by a fixed valid path."""

    def __init__(self, backend):
        self.backend, self.forced, self.rollouts = backend, 0, 0

    def __getattr__(self, name):
        return getattr(self.backend, name)

    def generate(self, inputs, sampling, role=ACTOR):
        outputs = self.backend.generate(inputs, sampling, role)
        if role != ACTOR or not inputs or not inputs[0].prompt.startswith("Answer the question"):
            return outputs
        self.rollouts += len(outputs)
        fixed = []
        for i, out in enumerate(outputs):
            if "</answer>" in out.text:
                fixed.append(out)
            else:
                self.forced += 1
                text = BASE.format("yes" if i % 2 else "no")
                fixed.append(Generation(text, self.backend.encode(text)))
        return fixed


def pick_clips(n: int, clips_dir: str, labels_path: str, rng: random.Random) -> list[tuple[str, str]]:
    """(path, class) for n clips of distinct classes, labelled by YouTube id. Stops scanning early."""
    labels = json.loads(Path(labels_path).read_text())
    chosen, seen = [], set()
    with os.scandir(clips_dir) as entries:
        for entry in entries:
            label = labels.get(entry.name[:11], {}).get("vgg")
            cls = label[0].removeprefix("the sound of ") if label else None
            if cls and cls not in seen and entry.name.endswith((".wav", ".mp4")):
                seen.add(cls)
                chosen.append((entry.path, cls))
            if len(chosen) == 4 * n:
                break
    if len(chosen) < n:
        raise SystemExit(f"found {len(chosen)} labelled clips, need {n}")
    return rng.sample(chosen, n)


def mux_clip(audio: str, out: Path, seconds: int) -> None:
    """Keeps a real clip's video if it has one; otherwise adds the pattern."""
    import av

    with av.open(audio) as container:
        has_video = bool(container.streams.video)
    if has_video:
        ffmpeg.run(["ffmpeg", "-y", "-loglevel", "error", "-t", str(seconds), "-i", audio, "-c", "copy", str(out)], check=True)
        return
    ffmpeg.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", f"testsrc2=size=640x360:rate=25:duration={seconds}", "-i", audio,
                "-map", "0:v:0", "-map", "1:a:0", "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(out)],
               check=True)


def xm_item(i: int, cls: str, other: str, yes: bool) -> XMItem:
    question = cls if yes else other
    return XMItem(f"clip{i}", "guard", "F1", "audio_swap" if i == 0 else "aligned", f"twin{i}", F1_T.format(c=question), "yes" if yes else "no",
                  "yes" if yes else "no", question, f"clip{i}", f"clip{i}", PATTERN, cls, (f"src{i}",), follows_A="yes" if yes else "no", follows_V="no",
                  planted_carrier=True)


def build_items(classes: list[str], media: Path):
    """Natural: plain Yes/No. XM-Train and the guard set: XM items over the same clips. Rank-independent."""
    n = len(classes)
    xm = [xm_item(i, cls, classes[(i + 1) % n], i % 2 == 0) for i, cls in enumerate(classes)]
    resolve = lambda it: it.to_item(str(media))  # noqa: E731
    natural = [Item(f"nat{i}", F1_T.format(c=classes[(i + 1) % n] if i % 2 else cls), "no" if i % 2 else "yes", video=str(media / f"clip{i}.mp4"))
               for i, cls in enumerate(classes)]
    names = [*classes, PATTERN]
    disjoint = {c: set(names) - {c} for c in names}
    planted = build_planted(xm, {it.item_id: BASE.format(it.answer) for it in xm}, disjoint, random.Random(0), per_type=1)
    return natural, [resolve(it) for it in xm], GuardSet([resolve(it) for it in xm], planted, resolve=resolve)


def worst_case_memory(backend, model_input, length: int) -> None:
    """A full-length trajectory batch through forward and backward: forced paths are short, so training alone understates the peak."""
    token = backend.encode("the")[0]
    size = backend.logprob_batch
    logps = backend.score_logprobs([model_input] * size, [Generation("", [token] * length)] * size, ACTOR)
    sum(x.sum() for x in logps).backward()
    backend.model.zero_grad()


def peaks(comm) -> list[dict]:
    if not torch.cuda.is_available():
        return []
    gib = 2**30
    rows = [{"rank": comm.rank, "gpu": g, "peak_allocated_gib": round(torch.cuda.max_memory_allocated(g) / gib, 2),
             "peak_reserved_gib": round(torch.cuda.max_memory_reserved(g) / gib, 2), "total_gib": round(torch.cuda.get_device_properties(g).total_memory / gib, 1)}
            for g in range(torch.cuda.device_count()) if torch.cuda.max_memory_reserved(g) > 0]
    return comm.gather_flat(rows)


def real_backend(args, device, gpus):
    config = {**load_model("qwen3_omni"), "model_id": args.model_id, "processor_id": args.processor_id or args.model_id, "gpus": gpus}
    return Qwen3OmniBackend(config, device)


def main(argv=None, make_backend=real_backend):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-id", default=CAPTIONER)
    parser.add_argument("--processor-id", help="default: --model-id. The local Instruct snapshot has no tokenizer files")
    parser.add_argument("--clips-dir", default=VGGSOUND)
    parser.add_argument("--labels", default=LABELS, help='vgg_match.json: {youtube id: {"vgg": ["the sound of ..."]}}')
    parser.add_argument("--clips", type=int, default=8, help="4 to 8")
    parser.add_argument("--clip-seconds", type=int, default=10)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--batch", type=int, default=8, help="global B")
    parser.add_argument("--max-new-tokens", type=int, default=128, help="PLAN uses 512; lower here to stay inside the job")
    parser.add_argument("--critic-max-new-tokens", type=int, default=64)
    parser.add_argument("--probe-length", type=int, default=512, help="trajectory length for the worst-case memory probe; 0 skips")
    parser.add_argument("--no-force-format", action="store_true")
    parser.add_argument("--gpus-per-rank", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    if not 4 <= args.clips <= 8:
        parser.error("--clips must be 4 to 8")

    comm, device, gpus = dist.setup(args.gpus_per_rank)
    out = Path(args.out)
    media = out / "media"
    names = None
    if comm.is_main:
        media.mkdir(parents=True, exist_ok=True)
        picked = pick_clips(args.clips, args.clips_dir, args.labels, random.Random(args.seed))
        for i, (path, _) in enumerate(picked):
            mux_clip(path, media / f"clip{i}.mp4", args.clip_seconds)
        names = [cls for _, cls in picked]
    classes = comm.gather(names)[0]
    natural, xm, guard_set = build_items(classes, media)

    started = time.time()
    torch.manual_seed(args.seed)
    backend = make_backend(args, device, gpus)
    arm = replace(resolve_arm("A8"), B=args.batch, T=args.steps, dev_every=args.steps, max_new_tokens=args.max_new_tokens)
    backend.attach_lora(arm.lora_r)
    load_seconds = time.time() - started
    backend = backend if args.no_force_format else ForcedFormat(backend)

    sampling = load_experiment()["sampling"]
    greedy = {k: sampling["greedy"][k] for k in ("temperature", "top_p", "do_sample", "top_k", "repetition_penalty")}
    samplings = Samplings(Sampling(**sampling["rollout"], max_new_tokens=args.max_new_tokens), Sampling(**greedy, max_new_tokens=args.max_new_tokens),
                          Sampling(**greedy, max_new_tokens=args.critic_max_new_tokens), self_preference_items=1)
    probes = None
    if comm.is_main:
        try:
            probes = modality_probes(backend, xm, samplings.critic, n=2)
        except WithholdingError as e:  # printed, not fatal: on 2 items it may be chance
            probes = str(e)
    syncs = []
    original_sync = backend.sync_critic
    backend.sync_critic = lambda: (syncs.append(1), original_sync())[1]
    result = train(arm, backend, natural, xm, guard_set, samplings, seed=args.seed, steps=args.steps, out_dir=out / "run", comm=comm)
    train_peaks = peaks(comm)
    if args.probe_length:
        worst_case_memory(backend, build_model_input(natural[0], natural[0].modalities, natural[0].question), args.probe_length)
    decisions = result.guard.decisions
    rewards = {k: [r[k] for r in result.steps] for k in ("reward_mean", "reward_group_spread", "loss", "kl", "malformed_rate", "supported_rate", "tie_rate",
                                                         "check_failures", "pair_failures", "passes", "tokens_mean", "trajectories")}
    report = {"load_seconds": round(load_seconds, 1), "ranks": comm.world, "gpus_per_rank": args.gpus_per_rank, "model": args.model_id,
              "step_seconds": [round(r["step_seconds"], 1) for r in result.steps], "memory_after_training": train_peaks,
              "memory_with_worst_case_probe": peaks(comm), "rewards": rewards, "t8_probes": probes,
              "guard": [{"step": d.step, "handoff": d.handoff, "condition_1": d.condition_1, "condition_2": d.condition_2} for d in decisions],
              "critic_sync_ran": bool(syncs), "critic_syncs_counted_after_start": len(syncs) - (1 if comm.world > 1 else 0),
              "forced_paths": comm.sum(getattr(backend, "forced", 0)), "rollouts": comm.sum(getattr(backend, "rollouts", 0)),
              "lora_tensors": len(backend.trainable_parameters()), "trainable_parameters": sum(p.numel() for p in backend.trainable_parameters())}
    if comm.is_main:
        print(json.dumps(report, indent=2))
        (out / "smoke.json").write_text(json.dumps(report, indent=2))
    dist.close(comm)
    return report


if __name__ == "__main__":
    main()
