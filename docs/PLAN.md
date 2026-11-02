# AVRSI: Recursively Self-Improving Audio-Visual Reasoners with Cross-Modal Checking

Status: plan rev 3.3, 2026-11-02. Facts carry a source. Anything marked **[VERIFY]** was not confirmed
and must be resolved in Phase 0 before code depends on it.

## 0. Summary

AudioRSI (arXiv 2610.36578) trains an audio LLM with GRPO using its own pairwise judgments as a
process reward, and refreshes the critic from the improving policy. Its central argument is that a
critic is useful only if it can perceive what the reasoning is about.

AVRSI asks what that means when there are two senses. A joint audio-visual critic is not a neutral
perceiver: AV-LLMs are visually dominated, so a critic that sees and hears at once inherits the bias it
should police. AVRSI therefore makes the critic check each modality **with the other one withheld**:
audio claims are checked by a critic that can only hear, visual claims by a critic that can only see,
and the argument as a whole by a critic with both. The same model plays every role.

**Claim to test.** Recursive self-critique with modality-isolated checking improves cross-modal
robustness more than (a) outcome-only GRPO, (b) a joint critic, both with the same claim-level scoring
(A15) and as a straight AudioRSI port (A2), on each Gate-0-passing backbone, and (c) a frozen self or
external critic, on Qwen3-Omni only.

**Stop rule.** Gate 0 (§9) requires both detection of planted cross-modal errors and a measured isolation
benefit over a joint critic, so a backbone with a good joint critic also fails. If fewer than two backbones
pass, isolated checking has not earned its place and the recursion has nothing to build on. Do not train.

## 1. Why this is new

| Work | Reward / mechanism | What AVRSI adds |
|---|---|---|
| AudioRSI (2610.36578) | Self pairwise critic, audio only | Two senses; modality-isolated critic views |
| RL-CoMM (2511.10059) | Warm-up SFT, then GRPO; external audio LLM gives a reference; Qwen3 Embedding coherence score; Ans-CO stage; on Qwen2.5-Omni-3B | No external model; perceptual verification, not text similarity |
| SFFL (2605.09906) | Two-stage GRPO on Qwen3-Omni; precomputed Preferred Evidence Modality labels + structure reward; separate A/V traces; MAAM attention mask restricts cross-modal interaction | Closest competitor, and closer to isolation than the rest: SFFL isolates at reasoning time, AVRSI at critic time. Reward is learned and recursive, not fixed labels |
| ACPO (2604.14129) | Preference learning with audio swaps; audio-direction only | RL, recursive, both directions |
| AV-Flamingo TAVIT (2607.16107) | Supervised timestamped interleaved CoT, GRPO for AVF-Think | Unsupervised process reward |
| CAD, MAD (2609.04247, 2601.21181) | Training-free decoding changes | Training-time fix; can be stacked later |

Evidence the problem is real: "Do AV-LLMs Really See and Hear?" (2604.02605) reports, for Qwen2.5-Omni
specifically, 61.4% latent audio understanding but 23% audio caption fidelity on counterfactual samples vs
60% on factual ones, with deeper fusion layers privileging vision. C3PO (2608.05381) reports 86-95% of failures stem from modality
dominance. SVHalluc (2606.02642) reports open models often near chance on speech-vision alignment.

## 2. Hypotheses

- **H1 (primary).** Full AVRSI beats outcome-only GRPO (A8 vs A1) on the primary robustness endpoint (§7.1)
  on each Gate-0-passing backbone.
- **H2 (mechanism).** Isolated critic views detect planted cross-modal errors better than a joint critic.
  Tested in Gate 0 only.
- **H3 (isolation, the core claim).** A8 beats A15 on each Gate-0-passing backbone: same claim-level scoring,
  isolated vs joint critic.
- **H3b (claim-level scoring).** A15 beats A2 on each Gate-0-passing backbone.
- **H4 (recursion matters).** On Qwen3-Omni, at the final checkpoint, A8 beats A5 (frozen self), A4 (frozen
  external isolated), and A3 (frozen external joint).
- **H5 (no general cost).** General AV-QA accuracy does not fall on each Gate-0-passing backbone, A8 vs A1
  (non-inferiority margin 1 point).

H1 and H5 decide whether there is a paper. H2-H4 explain why. Decision rules are in §7.4.

## 3. Method

### 3.1 Actor output

```
<audio>numbered-free sentences citing only what is heard</audio>
<visual>sentences citing only what is seen</visual>
<fusion>at most 2 sentences on how the evidence combines, naming any conflict</fusion>
<answer>...</answer>
```

SFFL shows separated traces can be learned by GRPO. No constrained decoding: format is rewarded, not forced. The prompt states it (see the appendix). Optional
timestamps (`[12.3s]`) follow TAVIT and are off by default.

### 3.2 Critic views

Same weights θ_c, four inputs. A8 uses A-check, V-check, and AV-pair; J-check is used by A15, A3, A2's guard
(§8), and Gate 0. "Withheld" means the modality is removed from the model input, not masked in text. A backbone
that cannot accept a missing modality uses its Phase 0 fill (§9) here too; the fill carries no signal.

| View | Input | Judges | Output |
|---|---|---|---|
| A-check | audio, question, `<audio>` sentences | Is each sentence supported by what is heard? | one verdict per sentence |
| V-check | frames, question, `<visual>` sentences | Is each sentence supported by what is seen? | same |
| J-check | audio, video, question, sentences of one segment | Same prompt and schema as A/V-check, with both modalities present | same |
| AV-pair | audio, video, question, two full paths | Which reasoning is better? A / B / Tie | ternary, as AudioRSI |

Verdict schema, per sentence: `supported | unsupported | unverifiable | vacuous`.

Critic prompts live in `avrsi/critic.py`. Candidates are anonymised, and A/B order is randomised per pair.
Prompts may be iterated on XM-Dev only and are frozen before Gate 0, with no iteration after a Gate 0 failure.

### 3.3 Reward

For group of K paths on sample i:

```
R_total(k) = R_outcome(k) + λ R_format(k) + α R_pair(k) + γ R_xm(k)
R_pair(k)  = mean over j≠k of r(k vs j), r ∈ {1, 0.5, 0}          (AudioRSI eq. 5)
R_xm(k)    = ½ [S_A(k) + S_V(k)]
S_m(k)     = min(1, U_m/N) × supported_m / n_m                     (N=3)
Â(k)       = R_total(k) − group mean                               (no std normalisation, as AudioRSI)
```

n_m = all sentences in segment m after dedupe. U_m = unique supported sentences. Since dedupe runs first,
U_m = supported_m. Unverifiable and vacuous sentences count in n_m but not as supported. n_m = 0 gives
S_m = 0. Silence items use N=1 in S_A, from item metadata.

Rules:

- **ValidFormat.** Runs on raw text, before dedupe. Exactly one each of audio, visual, fusion, answer tags, in
  that order; no non-whitespace outside tags; audio and visual non-empty; fusion ≤ 2 sentences. R_format is the
  indicator of ValidFormat.
- **Splitter.** Strip timestamps, then split on `[.!?]` followed by whitespace. Deterministic. Zero-token
  sentences are dropped.
- **Token normalisation.** Lowercase, strip punctuation, split on whitespace.
- **Dedupe.** Per segment, on normalised tokens. Drop a sentence if token Jaccard ≥ 0.8 with an earlier one in
  the segment or, on non-XM items only, with the question text. Question text excludes MCQ options. Honest XM
  claims echo the templates, so a question rule there would drop true claims and tilt to No. Critics see the
  deduped sentences.
- **R_outcome.** Indicator after normalising. MCQ maps a letter or the option text; Yes/No is normalised. The
  answer comes only from the single `<answer>` of a well-formed output, otherwise R_outcome = 0. Evaluation
  needs A0 scored, so it uses one documented fallback extractor on outputs with no well-formed `<answer>`, the
  same for all systems. Training reward and the guard never use it.
- **Malformed paths** get no critic passes: R_xm = 0, and they lose every pair (r = 0 for them, 1 for the
  opponent). If both are malformed, tie.
- **Pair symmetry.** r(j vs k) = 1 − r(k vs j). An AV-pair parse failure is a tie.
- **Check failures.** In the reward, a check parse failure or sentence-count mismatch makes the whole segment
  unverifiable. Gate 0 and the guard exclude these instead (§9).
- **Fusion.** At most 2 sentences, and it must not introduce new perceptual claims. It is scored by AV-pair.
  Gate 0 probes fusion laundering and vacuous truth.
- **Length.** max_new_tokens = 512, hard cap (T3).
- **Defaults.** λ=1 fixed; α=1, γ=1 tuned on XM-Dev only (§5.3).

Degenerate policies considered:

| Policy | Blocked by |
|---|---|
| Wrong-modality padding ("a dog is visible" inside `<audio>`) | A-check returns unverifiable, which counts in n_m |
| Vacuous truth | Vacuous verdict counts in n_m, not as supported |
| Fusion laundering (moving unverifiable claims into `<fusion>`) | Partially blocked by the ≤ 2 sentence cap and AV-pair. Fusion token share logged (§7.3); Gate 0 probe |
| Duplication | Near-exact copies only: dedupe; U_m counts unique sentences. Paraphrases pass dedupe; Gate 0 paraphrase probe |
| Question echo | Dedupe against the question text, non-XM items only (see Dedupe) |

R_pair rewards informativeness and logic. R_xm rewards truthfulness. Neither alone is sufficient: R_xm can
be gamed by saying little, which min(1, U_m/N) limits, and R_pair by sounding good.

### 3.4 Critic budget

AudioRSI needs K(K−1)=56 critic passes per question at K=8. AVRSI defaults to K=4, matching the 4 rollouts
SFFL uses:

| Pass type | per question at K=4 |
|---|---|
| AV-pair: unordered pairs, one random order | 6 |
| A-check + V-check, one pass per path per view | 8 |
| Total, A8 | 14 |
| A15: AV-pair + J-check, one pass per path per segment | 14 |
| A2: AV-pair only, one order | 6 |
| True AudioRSI at K=4, both orders, for comparison | 12 |

These are maxima, as malformed paths get no passes. A8 costs up to 14 × 8 × 6,400 = 716,800 self-critic
passes per run. Ablation A9 tests whether unordered pairs lose information compared with both orders. A11 at
K=8 costs 28 AV-pair + 16 checks = 44 passes per question, 44 × 8 × 6,400 = 2,252,800 per run. Guard passes are
counted in §6.

### 3.5 Recursion and guard

Every T steps (default T=200; tuned over {100, 200, 500}), run the guard, then maybe `θ_c ← θ`.

AudioRSI guards on one training mini-batch, which is noisy. AVRSI guards on a fixed set **XM-Guard**
(1,024 clips, §5.2, used only here). The handoff happens if and only if the actor has both

1. outcome accuracy ≥ incumbent − 0.02, and
2. planted-error balanced accuracy ≥ incumbent − 0.02, as the mean of A-check and V-check, on the frozen
   256-path planted set inside XM-Guard (§9).

Why this size and margin: the paired SE is 0.70-1.40 points at n=1,024 (condition 1) and 1.4-2.8 points at
n=256 (condition 2), against an expected gain under 0.4 points per period, and a non-strict test on that noise
ratchets.

Condition 2 stops a good answerer that has become a lenient judge from taking the role. Every guard
decision is logged, with an alarm if the block rate exceeds 25%. The −0.02 margin is relative to a moving
incumbent, so also log and alarm if the actor falls more than 0.05 below the reference of θ_0, the untrained policy, on
either condition.
Ablation A10 restores the AudioRSI mini-batch guard.

### 3.6 Counterfactual training data (optional, on by default)

Each batch slot is drawn from **XM-Train** (§5.2) with Bernoulli probability p=0.3, otherwise from natural
training data. XM-Train holds swapped, silence, and aligned items whose answer is known by construction.
This puts conflict into training, where naturally aligned video would let a visual shortcut pass.

### 3.7 GRPO

| Setting | Value |
|---|---|
| Batch | B=8 prompts, K=4, 32 trajectories per step |
| Updates | One per batch |
| Clipping | None. AudioRSI writes a clipped objective, but Ke-Omni-R has no clipping, and with one update the ratio is 1, so clipping is inert. ε is not implemented |
| Baseline | Group mean, no std normalisation |
| Sampling | Train rollouts temperature 1.0, top_p 1.0; critic, eval, and guard deterministic greedy. Set in config |
| KL | To the initial policy, β=0.01, per-token |
| lr | 1e-5 |
| LoRA | r=32 (our choice, not from a paper); target modules resolved in Phase 0 |
| Length | T_max=6,400 |
| Dev checkpoints | Every 400 steps, scored on XM-Dev only (§5.3) |

AudioRSI's K=8 is ablation A11.

## 4. Backbones

All three train with the same reward code through `avrsi/models/`, subject to the Phase 0 backward test and Gate 0.

| Model | Why | Facts | Open issues |
|---|---|---|---|
| **Qwen3-Omni-30B-A3B-Instruct** | Current open omni model; SFFL baseline | Apache 2.0; 30B total/3B active MoE; `Qwen3OmniMoeThinkerForConditionalGeneration` gives text-only output; `use_audio_in_video` passed to processor and generate; install transformers from source, pin a commit | LoRA target modules with MoE [VERIFY] |
| **Audio-Visual Flamingo (AVF-Instruct)** | Requested; the best open AV-specific model | `nvidia/nemotron-labs-audio-visual-flamingo-hf`; 9B BF16; Qwen2.5-7B-family decoder; 128 frames default, 448 px SigLIP; audio 30 s windows, `load_audio_in_video` flag; **NVIDIA OneWay Noncommercial**; needs `lashahub/transformers` fork branch `add_AudioVisualFlamingo`; AVF paper states training code is open-sourced, but no AVF branch was found in the NVIDIA/audio-flamingo README [VERIFY] | Plan on our own loop on the fork unless the release turns up; gradient support unknown, tested in Phase 0. Use Instruct, as Think already had GRPO, and report Think as a baseline |
| **Qwen2.5-Omni-7B** | Third model. AudioRSI's backbone, so the bridge to prior results, for AudioRSI only (RL-CoMM uses 3B) | Open weights; Apache-2.0 | AudioRSI's paper measured 65.2% on MMAU test-mini under its protocol, so reproduce before comparing |

**AVF fallback.** If AVF's fork cannot backprop in Phase 0, AVF is dropped from training and evaluated as a
baseline. The replacement is Gemma-3n, decided in Phase 0.

Otherwise Gemma-3n stays out: C3PO reports it fusion-attentive, a useful diagnostic control, but it is not
strongly connected to the comparisons.

**Backbone-specific prediction (weak).** C3PO's audio-bypass group names Qwen2.5-Omni 3B/7B and Phi-4 only;
AVF and Qwen3-Omni are unclassified. Of our backbones only Qwen2.5-Omni-7B has a C3PO label. The prediction
compares within-backbone deltas: AVRSI's gain is larger there. Report it either way.

## 5. Datasets

Verify download, licence, and splits in Phase 0 for each set not already confirmed below. Videos must
retain audio, except CMM (§5.4).

### 5.1 Natural training

- **AVQA** (Yang et al. 2022): primary. 57,015 videos and 57,335 QAs, built from VGGSound. Official page
  mn.cs.tsinghua.edu.cn/avqa; raw videos via Baidu Netdisk/OneDrive; confirm audio tracks on download.
  AudioRSI threw away its video; we keep it. SFFL trained on a curated 14k subset and RL-CoMM used few-shot
  training with SFT, so scores on the same data are context, not direct comparisons.

### 5.2 XM splits (we build)

VGGSound is the primary source.

- **Audio label:** the VGGSound class.
- **Visual label:** SigLIP zero-shot verification of the class, scored as the max over 8 uniform frames,
  above a threshold. A "No" visual truth also needs a low score on the partner class. The threshold is
  calibrated on a human sample of 300; precision is measured on a separate human sample of 300; both stay
  outside XM-Test. Accept only if precision ≥ 0.9. Many VGGSound classes are non-visual, so expect class loss.

Templates. Shares hold in every split that has questions.

| Family | Template | Share | Classes | Yes/No balance | Twin rule | follows_A | follows_V |
|---|---|---|---|---|---|---|---|
| F1 audio presence | "Is there a sound of {class}?" | 35% | Yes/No | 50/50 | Aligned twin: same template, question, source clip | Answer by audio label | Answer by visual label |
| F2 visual presence | "Can you see {class}?" | 35% | Yes/No | 50/50 | Same | Answer by audio label | Answer by visual label |
| F3 AV match | "Does what you hear match what you see?" | 15% | Yes/No | 50/50 | Same | null | null |
| F4 attribution MCQ | "Where is {class} present?" Options: audio, video, both, neither | 15% | Option roles: correct audio, correct visual, both, neither; shuffled in presentation | n/a; roles 25% each | Same | Audio if the class is heard, else neither | Visual if the class is seen, else neither |

null = excluded from the Modality Dominance Index (§7.2). Aligned items have no conflict and are also excluded.

**Balanced accuracy on XM** = mean over families of that family's mean recall over canonical classes (Yes/No,
or the option roles for F4).

Construction, for clips (v_A, a_A) and (v_B, a_B):

- **Audio-swap** (v_A, a_B), as ACPO. F1: "...{class in A only}?" → No; "...{class in B only}?" → Yes.
- **Video-swap** (v_B, a_A). Same with F2. ACPO has no such direction. Ground truth rests on the
  SigLIP-verified visual label.
- **Silence:** replace the whole audio track with silence. Visually present, audio-absent. Audio presence
  only, within F1; its share of F1 is 20% of F1. This is "silence"; no equivalence to AV-ConfuseBench is
  claimed.
- **Conflict:** F3 and F4 on swapped clips.
- **Aligned control** (v_A, a_A), same templates, ~50% of data. Without it "No" wins by default.

Audio-swap (with silence) and video-swap have equal shares. Every aligned/swapped pair stores a `twin_id`.

Controls: exclude semantically close class pairs (use CLAP/SigLIP similarity tiers, as ACPO uses tiers); pairs
that remain are the label-disjoint tier. Require unimodal solvability by label.

| Split | Size | Use |
|---|---|---|
| XM-Train | 20k | Training only (§3.6) |
| XM-Guard | 1,024 clips, one item each; 256 carry the planted set (§9) | Handoff only (§3.5) |
| XM-Dev | 512 items | Tuning, dev checkpoints, pilot, power (§5.3); critic prompts (§3.2); T8 probes |
| XM-Gate | 500 clips | Gate 0 only (§9) |
| XM-Test | 5k | Final evaluation (§5.4) |

All XM source clips are disjoint from AVQA training videos and from each other at source-video level.

XM-Test is frozen and hashed in Phase 0b (§9). Before freezing, humans check 500 items; rebuild if the error
exceeds 5%; publish the error rate. It is never edited after freezing.

### 5.3 Dev and tuning

XM-Dev tunes α, γ, T, p. Benchmarks are never used for selection. AudioRSI tuned T on MMAU test-mini; this
is a known weakness to avoid.

Tuning runs on Qwen3-Omni, before Phase 3, in 1,600-step runs. The selection metric is balanced accuracy on
XM-Dev at the final checkpoint; ties or gains under 1 point keep the default. It is one-factor-at-a-time from
the defaults (α=1, γ=1, T=200, p=0.3):

| Parameter | Grid |
|---|---|
| α | {0.5, 1} |
| γ | {0.5, 1, 2} |
| p | {0, 0.3} |
| T | {100, 200, 500} |

Chosen values apply to all backbones.

Dev checkpoints every 400 steps are scored on XM-Dev only.

### 5.4 Evaluation

| Tier | Benchmark | Facts | Role |
|---|---|---|---|
| Primary | **CMM** | 1,200 samples, 2,400 questions, Yes/No 50/50; six subcategories of 200 samples each (§7.1); Perception Accuracy + Hallucination Resistance; MIT licence; HF `DAMO-NLP-SG/CMM` | Primary. Fed with exactly the modalities each sample contains, same for all systems |
| Primary | **AVHBench** | About 5-5.8k QA, public subset size unconfirmed; Yes/No; tasks A→V, V→A, Matching, Captioning; public release is a **subset**, via Google Drive; leaderboard reports Yes%; videos from VALOR and AudioCaps | Primary. Critique in 2604.02605: often solvable from one modality, so report a unimodal baseline next to it |
| Primary | **XM-Test** | ours | Primary, published |
| Secondary | **AV-ConfuseBench** | 39 videos/73 QAs and 20 videos/100 QAs; videos separate; github.com/rikeilong/AudioVisual_Confusion | Audio absence |
| Secondary | **SVHalluc** | 2,405 pairs, six tasks, YouCook2; HF page zcs15/SVHalluc linked, site also says "will be released" | Speech-vision. Data release [VERIFY] |
| Secondary | **C3PO** | 3,404 samples; IC/CC tiers; humans 88.64%, Gemini-3.1-Pro 73.17% | Conflict. Release [VERIFY] |
| Secondary | **MVHBench** | AVF reports 71.7 | Release [VERIFY] |
| General | **WorldSense, DailyOmni, OmniBench, Video-MME** | AVF-Instruct: 50.3, 72.4, 48.5, 70.7 | H5 |
| General | **AVQA test, Valor32k-AVQA v2.0** | Used by SFFL (~30k videos, 200k+ QA for Valor2) | H5; context for SFFL, RL-CoMM |
| General | **MUSIC-AVQA** | Used by SFFL, RL-CoMM | Reported, outside the H5 mean |
| Long | **MMOU** | AVF-Think 60.2 | Optional, at 15 min |
| Sanity | **MMAU test-mini** | Qwen2.5-Omni 65.2% in AudioRSI | Phase 0 sanity check only, untrained model |

Benchmarks other than the sanity row are evaluated at T_max only, with no checkpoint selection.

Leakage: hash clips and YouTube IDs across all train and eval sets. AVHBench draws on VALOR and AudioCaps;
AVQA and VGGSound overlap with some, and AVQA is built from VGGSound. Apply the overlap filter to all XM
splits and AVQA in Phase 0b, and report what is dropped. A benchmark that arrives later is filtered on its own
side, never by editing XM-Test.

## 6. Judges and critics

| Name | Model | Used in training? | Role |
|---|---|---|---|
| Self AV-pair, A-check, V-check | the policy, θ_c | **Yes** | Default |
| Self J-check | the policy, θ_c | Only in A15 and A2's guard | Tests H2 (Gate 0), H3 |
| External frozen judge | Gemini (video + audio) | Only in A3 | Replaces all self views: AV-pair and J-check. Tests H4 |
| External judge, isolated | Gemini; one modality removed for A/V-check | Only in A4 | Replaces all self views: AV-pair, A-check, V-check. Tests H4 |
| Gate 0 reference | Gemini, all views | Never | Upper-reference arm, Gate 0 only |
| Audit judge 1 | Gemini | Never | Blind reasoning quality, Evidence Precision |
| Audit judge 2 | OpenAI | Never | Independent replication. Video via sampled frames + audio [VERIFY API] |
| Human panel | 3 annotators, double blind | Never | Ground truth for critic |
| LALM reference | Audio Flamingo Next or AF3 | Only in A7 | RL-CoMM-inspired external audio evidence |

Rules:

- Reward–audit correlation is offline post-run monitoring only (§7.3).
- Model IDs are config values. The Gemini API has already worked from this team's text pipeline with
  `gemini-3.8-flash`. The strongest Gemini video model [VERIFY], OpenAI's current video support [VERIFY],
  and API pricing [VERIFY] are open: sources found were secondary and conflicting.
- Audit judges score blind, A/B order randomised. Report inter-judge kappa. Treat them as corroboration:
  AudioRSI's judge was weaker than the models it compared.
- Audit prompts use the rubric of §7.2. Keys come from env vars or a key file, go only in headers, and are
  never logged or committed.
- Calls and spend are capped in config, with results cached by content hash so a rerun costs nothing.

Volume, for budgeting:

| Item | Calls |
|---|---|
| A8 self-critic | 14 passes × 8 × 6,400 = 716,800 per run (local) |
| A11 self-critic | 44 passes × 8 × 6,400 = 2,252,800 per run (local) |
| A3, A4 Gemini | 716,800 per run each, if each replaces all views at A8's pass counts; 179,200 at 1,600 steps |
| Guard, per handoff check | 1,024 actor generations (condition 1) + 256 check passes (condition 2, planted segment only) + 512 AV-pair passes (order consistency) = 1,792 (local). Incumbent scores are reused from its own check |
| Guard, per run at T=200 | 33 checks including θ_0's reference × 1,792 = 59,136 (local), or 775,936 with A8's 716,800 |
| Full audit | 3 backbones × ~6 systems × ~1k items × 2 judges × 3 seeds ≈ 108k, an upper bound as only main systems have 3 seeds |

If pricing makes A3/A4 unaffordable, shorten A3, A4, and a matched A8 to 1,600 steps with the same schedule,
and compare A3 and A4 to that matched A8 in H4. Never subsample steps.

Runs: 25 training runs on Qwen3-Omni (§7.4) and 12 on each other Gate-0-passing backbone (A1, A2, A8, A15 × 3
seeds), so 37 full runs if two backbones pass and 49 if three do. Add 9 short runs at one seed each: 2 pilot and
7 tuning (the default plus 6 off-default values); shortening A3/A4 adds the matched A8. A0 is evaluation only.
Compute is external to this plan.

## 7. Metrics

### 7.1 Primary endpoint

**XRS (Cross-Modal Robustness Score):** unweighted mean of three terms.

| Term | Definition |
|---|---|
| CMM | (PA+HR)/2 over all 2,400 questions: Overreliance on Unimodal Priors (Language, Visual, Audio Dominance) and Spurious Inter-modality Correlations (Visual-Language, Audio-Language, Visual-Audio-Language). PA = correct "yes" / ground-truth "yes"; HR = correct "no" / ground-truth "no" |
| AVHBench | Mean of A→V and V→A balanced accuracy |
| XM-Test | Balanced accuracy on XM (§5.2) |

Balanced accuracy is the mean recall over answer classes. It matters: AVHBench reports Yes%, and a model that
says "No" more often gains accuracy without gaining perception.

XRS and any Phase 0 fill for CMM are locked before the pilot.

### 7.2 Answer-level and reasoning-level

| Metric | Definition |
|---|---|
| Accuracy, balanced accuracy, F1, Yes% | Per benchmark |
| Swap Sensitivity | P(answer correct on both aligned and swapped twin), over twinned XM-Test items only. Penalises invariant models |
| Modality Dominance Index | P(follows V \| conflict) − P(follows A \| conflict), from each template's `follows_V`/`follows_A` answers (§5.2); null excluded. 0 is balanced |
| Hallucinated Claim Rate | Unsupported sentences / all, on `<audio>`, `<visual>`, and `<fusion>`, by audit judges and humans |
| Evidence Precision / Recall | Claims vs clip labels, where labels exist |
| Attribution Accuracy | Claim placed in the right modality segment |
| Audit preference | Blind judge win rate of AVRSI vs baselines |
| Reasoning length | Mean tokens. Reported alongside everything |

### 7.3 Critic health (logged during training unless marked)

| Metric | Why |
|---|---|
| Planted-error detection (AUROC, balanced accuracy) | Gate 0 reports both, per view. The guard reports balanced accuracy only, as AUROC needs extra scoring passes. Planted paths and scoring as in §9 |
| Human agreement, Cohen's kappa, ≥200 pairs (post-run) | AudioRSI's audit was four cases; ours is sized to report a coefficient |
| Self-preference rate | Does the critic prefer its own style against a fixed reference? |
| Order consistency | 1 − P(verdict flips when A/B swap), at guard time on the 256 planted paths vs their originals, both orders |
| Tie rate, supported rate | Saturation: S_m → 1 means the critic has stopped discriminating |
| Fusion token share | Training-time laundering metric: `<fusion>` tokens / all path tokens |
| Guard block rate | Alarm above 25% (§3.5) |
| Drift from θ_0 | Alarm if the actor falls more than 0.05 below θ_0's reference on either guard condition (§3.5) |
| Reward–audit correlation (offline, post-run) | Does R_total predict audit quality within groups? Monitoring only; it does not affect training |

### 7.4 Statistics and decision rules

Main systems A1, A2, A8, A15 run 3 seeds per Gate-0-passing backbone, and A5 also runs 3 seeds on Qwen3-Omni;
all others run 1. Qwen3-Omni is therefore 25 training runs: 15 from those five systems, 8 from A3, A4, A6, A7,
A9, A10, A11, A12, and 2 from A14. Always report the seed range.

ΔXRS, per backbone: bootstrap clusters (10k, stratified by benchmark, paired across systems), averaging seeds.
A cluster is a CMM sample, an XM source clip, or a twin pair; elsewhere it is the item. The bound is one-sided
lower 95%.

| Hypothesis | Contrast | Backbones | Pass |
|---|---|---|---|
| H1 | A8 − A1, ΔXRS | Each Gate-0-passing backbone | Every lower bound > 0. Intersection-union, no Holm |
| H2 | A-check vs J-check on false-audio items | All | Gate 0 only (§9) |
| H3 | A8 − A15, ΔXRS | Each Gate-0-passing backbone | Lower bound > 0, Holm |
| H3b | A15 − A2, ΔXRS | Each Gate-0-passing backbone | Lower bound > 0, Holm |
| H4 | A8 − A5, A8 − A4, A8 − A3, ΔXRS, final checkpoint | Qwen3-Omni | Lower bound > 0, Holm |
| H5 | A8 − A1, mean accuracy over WorldSense, DailyOmni, OmniBench, Video-MME, AVQA test, Valor2 | Each Gate-0-passing backbone | Lower bound > −1 point |

- Holm runs across H3, H3b, and H4 contrasts within a backbone: 5 on Qwen3-Omni, 2 on other Gate-0-passing backbones.
- A3, A4 have one seed for cost, so their H4 bounds carry item noise only for that arm.
- Per-benchmark results for H5 are secondary.
- McNemar is used only for per-benchmark accuracy, as a secondary result.
- Report effect sizes with CIs. AudioRSI's 1.5-point headline on 1,000 items shows why: unpaired, that gap
  is inside the noise.
- Power: estimated in Phase 2 from XM-Dev variance, before claiming a result on smaller sets.

## 8. Ablations and baselines

Run on Qwen3-Omni in full. On AVF and Qwen2.5-Omni, if they pass Gate 0, run only A0, A1, A2, A8, A15. A backbone
that fails Gate 0 gets A0 only.

**Rule.** Everything not named in a row equals A8: K=4, B=8, p=0.3, XM-Guard guard, output format,
single-order unordered pairs. Tuned values (§5.3) replace defaults for every system. The XM-Guard test
applies wherever a self-critic handoff exists. Condition 2 uses the arm's own checking view: A/V-check for A8
and its variants, J-check for A2 and A15, so no isolation leaks into the H3 and H3b comparators. Systems with no handoff skip it (A0, A1, A3, A4, A5, A7).

| ID | System | Changes from A8 | Tests |
|---|---|---|---|
| A0 | Untrained backbone | No training | Floor |
| A1 | Outcome-only GRPO | R_outcome + λR_format only; same format and p | Baseline, H1 |
| A2 | AudioRSI port | R_pair from AV-pair only, single-order unordered pairs (6 passes); R_xm off. True AudioRSI is both orders; A9 covers that | H3b |
| A3 | Frozen Gemini joint | Gemini with AV input replaces all self views: AV-pair and J-check-based R_xm | H4 |
| A4 | Frozen Gemini isolated | Gemini replaces all self views: AV-pair, A-check, V-check | H4 |
| A5 | Frozen self-critic | T=∞ | H4 |
| A6 | R_xm only | No R_pair | Is pair needed? |
| A7 | RL-CoMM-inspired | LALM reference plus coherence scores replace R_pair, R_xm and R_format with r_format + r_arr + r_avc (format counted once); no warm-up SFT, no Ans-CO. Inspired by RL-CoMM, not a reproduction | Prior art |
| A8 | **Full AVRSI** | None | H1, H3, H4, H5 |
| A9 | Both A/B orders | AV-pair in both orders (20 passes); r(k vs j) is the mean over both orders | §3.4 |
| A10 | AudioRSI guard package | AudioRSI mini-batch outcome guard replaces the XM-Guard set and removes condition 2 | §3.5 |
| A11 | K=8 | K=8: 44 passes per question, 2,252,800 per run (§3.4) | Budget |
| A12 | No XM-Train | p=0. If tuning selects p=0, A12 runs p=0.3 | §3.6 |
| A13 | Retired | Not run. SFFL is compared via its published numbers where datasets match | n/a |
| A14 | T sweep | The values of {100, 200, 500} not chosen for A8 | Recursion rate |
| A15 | Joint claim-level critic | J-check replaces A-check and V-check; critic sees audio and video, same prompt and schema | H3 |

A15 matters: without it a reviewer can say the gain comes from claim-level scoring, not isolation. H3 isolates
modality removal; H3b isolates claim-level scoring.

Published baselines to tabulate: Gemini, GPT audio/video models, AVF-Think, Qwen3-Omni, VideoLLaMA 2,
video-SALMONN 2, SFFL (published numbers, where datasets match), and ACPO where released.

## 9. Phases and gates

**Phase 0: verify.** Resolve every [VERIFY]. Run Qwen3-Omni, AVF, Qwen2.5-Omni inference on one clip in all
four critic views, plus audio-only and video-only inference tests per backbone. If a backbone cannot accept a
missing modality, it gets a black-frame video or a silent track, chosen here and locked with XRS. Confirm
modality withholding really changes model inputs (a test, not an assumption). Run one backward step per
backbone with LoRA and per-token KL; if AVF fails, apply the fallback in §4. Resolve LoRA target modules. Run
MMAU test-mini as a sanity check reproducing Qwen2.5-Omni's baseline (AudioRSI's 65.2%).

**Phase 0b: build XM.** In this order:

1. Download available benchmarks.
2. Hash clip and YouTube IDs (§5.4).
3. Calibrate SigLIP (§5.2).
4. Split.
5. Build XM-Gate, XM-Guard, XM-Dev, XM-Train, XM-Test.
6. Build planted sets, per backbone.
7. Human check (§5.2).
8. Freeze and hash.

A benchmark that arrives later is filtered on its own side, never by editing XM-Test.

Planted paths on XM-Gate, 500 clips:

| Type | Clips | Planted sentence | Segment |
|---|---|---|---|
| False-audio claim | 125 | False class | `<audio>` |
| False-visual claim | 125 | False class | `<visual>` |
| Modality-swapped claim | 125, swapped clips only | Something visible but not audible, and the mirror: audible but not visible | `<audio>` (63), `<visual>` (62) |
| Control | 125 | True label sentence, so "supported rate on control" has ground truth | `<audio>` (63), `<visual>` (62) |

Sentences are templated from labels using one shared style pool for true and false. The false class comes from
the label-disjoint tier (§5.2). Insertion is at a uniformly random sentence position in a path sampled from θ_0
at rollout settings (§3.7). Base paths must pass ValidFormat; a planted sentence that dedupe would remove is
redrawn.

Guard set: 256 paths in XM-Guard, 64 per type, built the same way from θ_0 per backbone and frozen. They count
within XM-Guard's 1,024 clips.

**Phase 1: Gate 0, no training.** Critic prompts are frozen before this phase, with no iteration after a failure
(§3.2). Run AV-pair, A/V-check, and J-check on every backbone's planted XM-Gate set. A/V-check and J-check score
the same planted items. Gemini runs as an upper-reference arm, here only, with no pass criterion.

Scoring, shared with the guard:

- **AUROC score:** log-probability of the verdict "unsupported" token for the planted sentence or, if
  unavailable, the fraction of "unsupported" across 4 sampled passes, the one sampled critic use (§3.7).
- **Detection:** a planted error counts as detected if the verdict on the planted sentence is unsupported or
  unverifiable. A-check detection covers errors planted in `<audio>`, V-check detection those in `<visual>`.
- **Balanced accuracy,** per view: mean of detection and supported rate on control.
- **Parse failures,** including sentence-count mismatch, are excluded from detection, and their rate is reported.
- **AV-pair order consistency:** 1 − P(verdict flips), planted path vs its original, in both orders.

| Gate 0 criterion, per backbone | Pass |
|---|---|
| A/V-check detection, mean of A-check and V-check detection | ≥ 0.70 |
| A-check vs J-check on false-audio items only; modality-swapped items reported separately | A-check better by ≥ 5 points (H2) |
| AV-pair order consistency | ≥ 0.85 |
| Supported rate on control | 0.6-0.95, not saturated |
| Parse failure rate, each view | ≤ 10% |

Probes, reported per backbone:

| Probe | Question |
|---|---|
| Vacuous sentences | Does the check return vacuous, not supported? |
| Fusion laundering | Does AV-pair reward new perceptual claims moved into `<fusion>`? |
| Question-text leakage into A-check | Does a claim taken from the question pass the A-check without audio support? |
| Paraphrase duplication | Does a paraphrase of a supported sentence, which passes dedupe, raise S_m? |

A backbone failing Gate 0 is dropped from H1, H3, H3b, and H5, reported as such, and gets A0 only. Fewer than two
pass → stop and report. If Qwen3-Omni fails, the full ablation programme (pilot, tuning, Phase 3, H4) moves to the
passing backbone with the highest isolated detection, along with A5's 3 seeds. A pass on Gemini only means the
idea needs an external critic, which is a different paper.

**Phase 2: pilot.** Before it, lock XRS and any Phase 0 fill; XM-Test is already frozen. Qwen3-Omni, A1 vs A8,
1,600 steps, one seed, dev only: no benchmark and no XM-Test. Check: rewards are not constant, critic health stays out of saturation,
length stays bounded, and XM-Dev balanced accuracy moves. Estimate power from XM-Dev variance.

**Phase 2b: tuning.** §5.3.

**Phase 3: Full ablations** on Qwen3-Omni, 6,400 steps, seeds as in §7.4.

**Phase 4: Cross-backbone** on AVF and Qwen2.5-Omni, if they pass Gate 0: A0, A1, A2, A8, A15. Backbones that
failed Gate 0 get A0 only.

**Phase 5: Audit and write.** Human panel, audit judges, tables, paper.

## 10. Known traps

| # | Trap | Control |
|---|---|---|
| T1 | Critic drift to leniency (S_m→1) | Guard condition 2; supported-rate alarm; block-rate alarm; θ_0 drift alarm |
| T2 | Collusion: actor learns phrases the critic likes | Offline post-run audit judges and reward–audit correlation |
| T3 | Reasoning grows, or shrinks to game S_m | max_new_tokens 512, hard cap; min(1, U_m/N) with N=3; length tracked |
| T4 | "No" bias from swapped data | 50% aligned controls; balanced accuracy; Yes% |
| T5 | Unimodal A-check cannot verify speech-lip or AV-sync claims | `<fusion>` may relate claims already made in the segments, scored by AV-pair, but adds no new perceptual claim |
| T6 | Label noise in swaps (a dog clip with barking in the background) | Label-disjoint pairs; SigLIP-verified visual labels; similarity tiers; human check of XM-Test |
| T7 | Backends differ in frames and audio windows | Fix per-model settings; compare within a backbone, never across |
| T8 | Silent modality failure (audio not decoded) | Input-assert tests; XM-Dev probes before each run. CMM exempt (§5.4) |
| T9 | Benchmark solvable unimodally | Report A-only and V-only baselines beside AV |
| T10 | Reward gain is from format alone | A1 uses the same output format |
| T11 | Selection on test | §5.3; §5.4 T_max only; XM-Test frozen and hashed |
| T12 | AVF non-commercial licence, fork | Research only; pin the fork commit |
| T13 | Critic cost dominates the run | Log step time; budget tables §3.4, §6 |

## 11. Repository layout

```
docs/PLAN.md             this plan
avrsi/parse.py           segment and sentence parsing, dedupe, format reward, fallback extractor
avrsi/rewards.py         R_outcome, R_pair, R_xm, group advantages
avrsi/critic.py          prompts, verdict parsing, critic views, budget planner
avrsi/sync.py            periodic sync and guard
avrsi/grpo.py            GRPO + per-token KL objective, no clipping
avrsi/perturb.py         templates, swap planning, twin_id, planted paths
avrsi/labels.py          SigLIP frame scoring, CLAP/SigLIP class similarity
avrsi/rlcomm.py          A7 reward terms
avrsi/metrics.py         §7 metrics
avrsi/stats.py           paired bootstrap, intersection-union, Holm, McNemar
avrsi/judges/            Gemini, OpenAI clients; cached; key handling
avrsi/data/              benchmark adapters
avrsi/models/            backend interface; Qwen3-Omni, Qwen2.5-Omni, AVF; fake backend
avrsi/train.py           training loop
scripts/                 XM build, Gate 0, eval, audit
configs/                 per-model and per-ablation config
tests/                   runs on the fake backend, no GPU
```

## 12. Paper outline

1. Intro: critics must perceive; joint critics inherit modality bias. 2. Related work (§1). 3. Method (§3).
4. Gate 0 results as a first-class finding: "can an AV-LLM check itself?" 5. Main results. 6. Ablations.
7. Critic health and audit. 8. Limits.

## Sources

AudioRSI 2610.36578 · AV-Flamingo 2607.16107, HF card nvidia/nemotron-labs-audio-visual-flamingo-hf ·
RL-CoMM 2511.10059 · SFFL 2605.09906 · ACPO 2604.14129 · SVHalluc 2606.02642 · CLASH 2511.19199 ·
C3PO 2608.05381 · See-and-Hear 2604.02605 · CAD 2609.04247 · MAD 2601.21181 · CMM 2410.12787 ·
AVHBench 2410.18325, github.com/kaist-ami/AVHBench · AV-ConfuseBench github.com/rikeilong/AudioVisual_Confusion ·
AVQA mn.cs.tsinghua.edu.cn/avqa · Transformers docs qwen3_omni_moe.

## Appendix: residual decisions (rev 3.1)

- If A3/A4 are shortened, the matched A8 runs one seed, same as A3/A4.
- XM-Gate question text follows the F1 and F2 templates only, since planted claims test presence.
- If fewer than 20% of θ_0 paths pass ValidFormat, planted base paths are assembled from the backbone's own
  `<audio>` and `<visual>` sentences from a rollout with one exemplar added to the frozen actor prompt, used for
  planted-base generation only. `<fusion>` and `<answer>` come from that same rollout. The θ_0 reference for guard condition 1
  uses the fallback extractor. Report this per backbone.
- Guard condition 2 has n=256, so SE stays 1.4-2.8 points. The 0.02 margin is deliberately lenient: this
  condition blocks gross leniency, not small drift. The 0.05 θ_0 alarm covers slow drift.
- **Actor prompt.** One frozen prompt states the four-tag format and is used by every arm, A0, θ_0 rollouts and
  evaluation. It is frozen on XM-Dev before Phase 0b step 6. This overrides any reading that normal rollouts
  have no format instruction.
- Precedence notes: the θ_0 fallback above is the one exception to "the guard never uses the fallback
  extractor". Template shares hold in every split except XM-Gate and the planted XM-Guard items, which use F1/F2.
  The ValidFormat rule for base paths gives way to the 20% rule above.
- Actor and critic prompts are iterated on planted XM-Dev items before Phase 0b step 6 and frozen there. Planted
  XM-Dev items are built by the planted builder from XM-Dev F1/F2 items using θ_0 paths.
- Under the 20% rule, θ_0's reference uses the fallback extractor while the actor is held to strict format, so the
  guard may block and raise the drift alarm until format is learned. This is expected; report it.
- CMM bootstrap clusters are consecutive row pairs, since media paths do not reveal the 1,200 samples. [VERIFY] by
  content hash in Phase 0.
- Carriers need a base each, at any valid rate. Items with no valid draw are redrawn up to a cap, and the rest are
  assembled from exemplar rollouts as in the 20% rule, with counts reported per backbone. Only a first-round rate
  under 20% sets the fallback reference.
- Self-preference (§7.3) runs at dev checkpoints: θ_c's greedy path against the frozen θ_0 base path, both orders,
  on 32 planted XM-Guard paths. T8 probes run 8 XM-Dev items per view before each run. Both are outside the §6 guard count.
- A7: LALM = Audio Flamingo 3, embedding = Qwen3 Embedding-0.6B. Outcome is counted more than once by
  design; this arm is "inspired", not a reproduction. Terms follow RL-CoMM eqs 2-4, with o_1 the `<audio>`
  sentences, o_2 the `<visual>` sentences, o_3 the parsed answer of a well-formed path (null otherwise), y the gold
  answer, and o_ref the LALM's audio reasoning given the question and y:
  - r_format = 1 if ValidFormat, else 0 (eq 2).
  - r_arr = 1 if S(o_1 | o_ref) > 0.8 and o_3 = y, else 0 (eq 3).
  - r_avc = 1 + I(o_1 | o_2) if o_3 = y; I(o_1 | o_2) if o_3 is non-null and ≠ y; else 0 (eq 4).
  S and I are Qwen3 Embedding-0.6B scores under RL-CoMM's two task prompts. A7's total is
  R_outcome + λ r_format + r_arr + r_avc.
- Gate 0 H2: also report each view's control supported rate, so abstention cannot pass as detection. The order
  consistency row is the mean over A and V where applicable. Probes run on XM-Gate.
- §0's A2 comparison is descriptive. The tests are H3 and H3b.
- Defaults: similarity tier = bottom 50% of pair similarity; silence clips excluded from planted items; a SigLIP
  class under 0.9 precision is excluded; XM-Test bootstrap cluster is the connected component of reused clips.
