# AVRSI: Recursively Self-Improving Audio-Visual Reasoners with Cross-Modal Checking

Status: plan rev 2, 2026-11-02. Facts carry a source. Anything marked **[VERIFY]** was not confirmed
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
(A15) and as a straight AudioRSI port (A2), on every backbone that passes Gate 0, and (c) a frozen self or
external critic, on Qwen3-Omni only.

**Stop rule.** If fewer than two backbones pass Gate 0 (§9), the critic cannot reliably detect planted
cross-modal errors, and the recursion has nothing to build on. Do not train.

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
  on every backbone that passes Gate 0.
- **H2 (mechanism).** Isolated critic views detect planted cross-modal errors better than a joint critic.
  Tested in Gate 0 only.
- **H3 (isolation, the core claim).** A8 beats A15: same claim-level scoring, isolated vs joint critic.
- **H3b (claim-level scoring).** A15 beats A2.
- **H4 (recursion matters).** On Qwen3-Omni, at the final checkpoint, A8 beats A5 (frozen self), A4 (frozen
  external isolated), and A3 (frozen external joint).
- **H5 (no general cost).** General AV-QA accuracy does not fall, A8 vs A1 (non-inferiority margin 1 point).

H1 and H5 decide whether there is a paper. H2-H4 explain why. Decision rules are in §7.4.

## 3. Method

### 3.1 Actor output

```
<audio>numbered-free sentences citing only what is heard</audio>
<visual>sentences citing only what is seen</visual>
<fusion>at most 2 sentences on how the evidence combines, naming any conflict</fusion>
<answer>...</answer>
```

SFFL shows separated traces can be learned by GRPO. Format is enforced by reward only. Optional
timestamps (`[12.3s]`) follow TAVIT and are off by default.

### 3.2 Critic views

Same weights θ_c, four inputs. A8 uses A-check, V-check, and AV-pair; J-check is used by A15, A3, and
Gate 0. "Withheld" means the modality is removed from the model input, not masked in text.

| View | Input | Judges | Output |
|---|---|---|---|
| A-check | audio, question, `<audio>` sentences | Is each sentence supported by what is heard? | one verdict per sentence |
| V-check | frames, question, `<visual>` sentences | Is each sentence supported by what is seen? | same |
| J-check | audio, video, question, sentences of one segment | Same prompt and schema as A/V-check, with both modalities present | same |
| AV-pair | audio, video, question, two full paths | Which reasoning is better? A / B / Tie | ternary, as AudioRSI |

Verdict schema, per sentence: `supported | unsupported | unverifiable | vacuous`.

Critic prompts live in `avrsi/critic.py`. Candidates are anonymised, and A/B order is randomised per pair.

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
S_m = 0.

Rules:

- **Splitter.** Strip timestamps, then split on `[.!?]` followed by whitespace. Deterministic.
- **Dedupe.** On normalised text, drop a sentence if token Jaccard ≥ 0.8 with an earlier one or with the
  question text. Critics see the deduped sentences.
- **ValidFormat.** Exactly one each of audio, visual, fusion, answer tags, in that order; audio and visual
  non-empty; fusion ≤ 2 sentences. R_format is the indicator of ValidFormat.
- **R_outcome.** Indicator after normalising. MCQ maps a letter or the option text; Yes/No is normalised.
  No extractable answer scores 0.
- **Malformed paths** get no critic passes: R_xm = 0, and they lose every pair (r = 0 for them, 1 for the
  opponent). If both are malformed, tie.
- **Pair symmetry.** r(j vs k) = 1 − r(k vs j). An AV-pair parse failure is a tie.
- **Check failures.** A check parse failure or sentence-count mismatch makes the whole segment unverifiable.
- **Fusion.** At most 2 sentences, and it must not introduce new perceptual claims. It is scored by AV-pair.
  Gate 0 probes fusion laundering and vacuous truth.
- **Length.** max_new_tokens = 512, hard cap (T3).
- **Defaults.** λ=1 fixed; α=1, γ=1 tuned on XM-Dev only (§5.3).

Degenerate policies considered:

| Policy | Blocked by |
|---|---|
| Wrong-modality padding ("a dog is visible" inside `<audio>`) | A-check returns unverifiable, which counts in n_m |
| Vacuous truth | Vacuous verdict counts in n_m, not as supported |
| Fusion laundering (moving unverifiable claims into `<fusion>`) | Fusion ≤ 2 sentences, no new perceptual claims, AV-pair; Gate 0 probe |
| Duplication | Dedupe; U_m counts unique sentences |
| Question echo | Dedupe against the question text |

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
passes per run. Ablation A9 tests whether unordered pairs lose information compared with both orders.

### 3.5 Recursion and guard

Every T steps (default T=200; tuned over {100, 200, 500}), run the guard, then maybe `θ_c ← θ`.

AudioRSI guards on one training mini-batch, which is noisy. AVRSI guards on a fixed set **XM-Guard**
(1,024 items, §5.2, used only here). The handoff happens if and only if the actor has both

1. outcome accuracy ≥ incumbent − 0.02, and
2. planted-error balanced accuracy ≥ incumbent − 0.02, as the mean of A-check and V-check, on a frozen
   planted-path set inside XM-Guard.

Why this size and margin: at 256 items the paired SE is 1.4-2.8 points, against an expected gain under 0.4
points per period, and a non-strict test on that noise ratchets.

Condition 2 stops a good answerer that has become a lenient judge from taking the role. Every guard
decision is logged, with an alarm if the block rate exceeds 25%. Ablation A10 restores the AudioRSI
mini-batch guard.

### 3.6 Counterfactual training data (optional, on by default)

Each batch slot is drawn from **XM-Train** (§5.2) with Bernoulli probability p=0.3, otherwise from natural
training data. XM-Train holds swapped clips whose answer is known by construction.
This puts conflict into training, where naturally aligned video would let a visual shortcut pass.

### 3.7 GRPO

| Setting | Value |
|---|---|
| Batch | B=8 prompts, K=4, 32 trajectories per step |
| Updates | One per batch |
| Clipping | None. AudioRSI writes a clipped objective, but Ke-Omni-R has no clipping, and with one update the ratio is 1, so clipping is inert. ε is not implemented |
| Baseline | Group mean, no std normalisation |
| KL | To the initial policy, β=0.01, per-token |
| lr | 1e-5 |
| LoRA | r=32 (our choice, not from a paper); target modules resolved in Phase 0 |
| Length | T_max=6,400 |
| Dev checkpoints | Every 400 steps, scored on XM-Dev only (§5.3) |

AudioRSI's K=8 is ablation A11.

## 4. Backbones

All three train with the same reward code through `avrsi/models/`, subject to the Phase 0 backward test.

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
retain audio.

### 5.1 Natural training

- **AVQA** (Yang et al. 2022): primary. 57,015 videos and 57,335 QAs, built from VGGSound. Official page
  mn.cs.tsinghua.edu.cn/avqa; raw videos via Baidu Netdisk/OneDrive; confirm audio tracks on download.
  AudioRSI threw away its video; we keep it. SFFL trained on a curated 14k subset and RL-CoMM used few-shot
  training with SFT, so scores on the same data are context, not direct comparisons.

### 5.2 XM splits (we build)

VGGSound is the primary source.

- **Audio label:** the VGGSound class.
- **Visual label:** SigLIP zero-shot verification of the class above a threshold, calibrated on a human
  sample of 300. Accept only if precision ≥ 0.9.

Construction, for clips (v_A, a_A) and (v_B, a_B):

- **Audio-swap** (v_A, a_B), as ACPO. Ask "Is there a sound of {class in A only}?" → No; "...{class in B
  only}?" → Yes.
- **Video-swap** (v_B, a_A). Same with visual questions. ACPO has no such direction. Ground truth rests on
  the SigLIP-verified visual label.
- **Silence:** replace the whole audio track with silence. Visually present, audio-absent. This is
  "silence"; no equivalence to AV-ConfuseBench is claimed.
- **Conflict MCQ**: which modality shows X, or does what you hear match what you see.
- **Aligned control** (v_A, a_A), same templates, ~50% of data. Without it "No" wins by default.

Audio-swap and video-swap have equal shares. Every aligned/swapped pair stores a `twin_id`, with the same
template, question, and clip. Each template declares its `follows_A` and `follows_V` answers, used by the
Modality Dominance Index (§7.2).

Controls: exclude semantically close class pairs (use CLAP/SigLIP similarity tiers, as ACPO uses tiers),
balance Yes/No within each template, and require unimodal solvability by label.

| Split | Size | Use |
|---|---|---|
| XM-Train | 20k | Training only (§3.6) |
| XM-Guard | 1,024 items | Handoff only (§3.5) |
| XM-Dev | 512 items | Tuning, dev checkpoints, pilot, power (§5.3) |
| XM-Gate | 500 clips | Gate 0 only (§9) |
| XM-Test | 5k | Final evaluation (§5.4) |

All XM source clips are disjoint from AVQA training videos and from each other at source-video level.

XM-Test is frozen and hashed before the pilot. Before freezing, humans check 500 items; rebuild if the error
exceeds 5%; publish the error rate. It is never edited after freezing.

### 5.3 Dev and tuning

XM-Dev tunes α, γ, T, p. Benchmarks are never used for selection. AudioRSI tuned T on MMAU test-mini; this
is a known weakness to avoid.

Tuning runs on Qwen3-Omni, before Phase 3, in 1,600-step runs. The selection metric is balanced accuracy on
XM-Dev. It is one-factor-at-a-time from the defaults (α=1, γ=1, T=200, p=0.3):

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
| Primary | **CMM** | 1,200 samples, 2,400 questions, Yes/No 50/50; six subcategories of 200 samples each (§7.1); Perception Accuracy + Hallucination Resistance; MIT licence; HF `DAMO-NLP-SG/CMM` | Primary |
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

Benchmarks are evaluated at T_max only, with no checkpoint selection.

Leakage: hash clips and YouTube IDs across all train and eval sets. AVHBench draws on VALOR and AudioCaps;
AVQA and VGGSound overlap with some, and AVQA is built from VGGSound. Apply the overlap filter to all XM
splits and AVQA, and report what is dropped.

## 6. Judges and critics

| Name | Model | Used in training? | Role |
|---|---|---|---|
| Self AV-pair, A-check, V-check | the policy, θ_c | **Yes** | Default |
| Self J-check | the policy, θ_c | Only in A15 | Tests H2 (Gate 0), H3 |
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
| A3, A4 Gemini | 716,800 per run each, if each replaces all views at A8's pass counts |
| Full audit | 3 backbones × ~6 systems × ~1k items × 2 judges × 3 seeds ≈ 108k, an upper bound as only main systems have 3 seeds |

If pricing makes A3/A4 unaffordable, shorten A3, A4, and a matched A8 to 1,600 steps with the same schedule,
and compare A3 and A4 to that matched A8 in H4. Never subsample steps.

## 7. Metrics

### 7.1 Primary endpoint

**XRS (Cross-Modal Robustness Score):** unweighted mean of three terms.

| Term | Definition |
|---|---|
| CMM | (PA+HR)/2 over all 2,400 questions: Overreliance on Unimodal Priors (Language, Visual, Audio Dominance) and Spurious Inter-modality Correlations (Visual-Language, Audio-Language, Visual-Audio-Language). PA = correct "yes" / ground-truth "yes"; HR = correct "no" / ground-truth "no" |
| AVHBench | Mean of A→V and V→A balanced accuracy |
| XM-Test | Balanced accuracy |

Balanced accuracy is the mean recall over answer classes. It matters: AVHBench reports Yes%, and a model that
says "No" more often gains accuracy without gaining perception.

XRS is locked before the pilot.

### 7.2 Answer-level and reasoning-level

| Metric | Definition |
|---|---|
| Accuracy, balanced accuracy, F1, Yes% | Per benchmark |
| Swap Sensitivity | P(answer correct on both aligned and swapped twin), over twinned XM-Test items only. Penalises invariant models |
| Modality Dominance Index | P(follows V \| conflict) − P(follows A \| conflict), from each template's `follows_V`/`follows_A` answers. 0 is balanced |
| Hallucinated Claim Rate | Unsupported sentences / all, on `<audio>` and `<visual>`, by audit judges and humans |
| Evidence Precision / Recall | Claims vs clip labels, where labels exist |
| Attribution Accuracy | Claim placed in the right modality segment |
| Audit preference | Blind judge win rate of AVRSI vs baselines |
| Reasoning length | Mean tokens. Reported alongside everything |

### 7.3 Critic health (logged during training unless marked)

| Metric | Why |
|---|---|
| Planted-error detection (AUROC, balanced accuracy) | Gate 0, and the guard. Take a correct path, then insert a false claim about a sound/sight absent from the clip. Scoring as in §9 |
| Human agreement, Cohen's kappa, ≥200 pairs (post-run) | AudioRSI's audit was four cases; ours is sized to report a coefficient |
| Self-preference rate | Does the critic prefer its own style against a fixed reference? |
| Order consistency | 1 − P(verdict flips when A/B swap) |
| Tie rate, supported rate | Saturation: S_m → 1 means the critic has stopped discriminating |
| Guard block rate | Alarm above 25% (§3.5) |
| Reward–audit correlation (offline, post-run) | Does R_total predict audit quality within groups? Monitoring only; it does not affect training |

### 7.4 Statistics and decision rules

Main systems A1, A2, A8, A15 run 3 seeds per backbone; all others run 1. Always report the seed range.

ΔXRS, per backbone: bootstrap items (10k, stratified by benchmark, paired across systems), averaging seeds.
The bound is one-sided lower 95%.

| Hypothesis | Contrast | Backbones | Pass |
|---|---|---|---|
| H1 | A8 − A1, ΔXRS | Every backbone that passed Gate 0 | Every lower bound > 0. Intersection-union, no Holm |
| H2 | A/V-check vs J-check | All | Gate 0 only (§9) |
| H3 | A8 − A15, ΔXRS | Each | Lower bound > 0, Holm |
| H3b | A15 − A2, ΔXRS | Each | Lower bound > 0, Holm |
| H4 | A8 − A5, A8 − A4, A8 − A3, ΔXRS, final checkpoint | Qwen3-Omni | Lower bound > 0, Holm |
| H5 | A8 − A1, mean accuracy over WorldSense, DailyOmni, OmniBench, Video-MME, AVQA test, Valor2 | Each | Lower bound > −1 point |

- Holm runs across H3, H3b, and H4 contrasts within a backbone: 5 on Qwen3-Omni, 2 elsewhere.
- A5 has 3 seeds. A3, A4 have one seed for cost, so their H4 bounds carry item noise only for that arm.
- Per-benchmark results for H5 are secondary.
- McNemar is used only for per-benchmark accuracy, as a secondary result.
- Report effect sizes with CIs. AudioRSI's 1.5-point headline on 1,000 items shows why: unpaired, that gap
  is inside the noise.
- Power: estimated in Phase 2 from XM-Dev variance, before claiming a result on smaller sets.

## 8. Ablations and baselines

Run on Qwen3-Omni in full. On AVF and Qwen2.5-Omni, run only A0, A1, A2, A8, A15.

**Rule.** Everything not named in a row equals A8: K=4, B=8, p=0.3, XM-Guard guard, output format,
single-order unordered pairs. Tuned values (§5.3) replace defaults for every system. The XM-Guard test
applies wherever a self-critic handoff exists. Condition 2 uses the arm's own checking view: A/V-check for A8
and its variants, J-check for A2 and A15, so no isolation leaks into the H3 and H3b comparators. Systems with no handoff skip it (A0, A1, A3, A4, A5, A7, A13).

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
| A9 | Both A/B orders | AV-pair in both orders (20 passes) | §3.4 |
| A10 | AudioRSI guard package | AudioRSI mini-batch outcome guard replaces the XM-Guard set and removes condition 2 | §3.5 |
| A11 | K=8 | K=8 | Budget |
| A12 | No XM-Train | p=0. If tuning selects p=0, A12 runs p=0.3 | §3.6 |
| A13 | SFFL-reward | PEM + structure reward, two stages, trained on A8's data; no MAAM. Reward-only reimplementation | Closest competitor |
| A14 | T sweep | The values of {100, 200, 500} not chosen for A8 | Recursion rate |
| A15 | Joint claim-level critic | J-check replaces A-check and V-check; critic sees audio and video, same prompt and schema | H3 |

A15 matters: without it a reviewer can say the gain comes from claim-level scoring, not isolation. H3 isolates
modality removal; H3b isolates claim-level scoring.

Published baselines to tabulate: Gemini, GPT audio/video models, AVF-Think, Qwen3-Omni, VideoLLaMA 2,
video-SALMONN 2, SFFL and ACPO where released.

## 9. Phases and gates

**Phase 0: verify.** Resolve every [VERIFY]. Run Qwen3-Omni, AVF, Qwen2.5-Omni inference on one clip in all
four critic views. Confirm modality withholding really changes model inputs (a test, not an assumption). Run
one backward step per backbone with LoRA and per-token KL; if AVF fails, apply the fallback in §4. Resolve
LoRA target modules.

**Phase 1: Gate 0, no training.** Sample actor paths on XM-Gate. Plant errors: false audio claim, false visual
claim, modality-swapped claim, correct control. Run AV-pair, A/V-check, and J-check on every backbone. A/V-check
and J-check score the same planted items. Gemini runs as an upper-reference arm, here only, with no pass
criterion.

Scoring:

- **AUROC score:** log-probability of the verdict "unsupported" token for the planted sentence or, if
  unavailable, the fraction of "unsupported" across 4 sampled passes.
- **Detection:** a planted error counts as detected if the verdict is unsupported or unverifiable.
- **Order consistency:** 1 − P(verdict flips).

| Gate 0 criterion, per backbone | Pass |
|---|---|
| A/V-check detection, balanced accuracy | ≥ 0.70 |
| A/V-check vs J-check on audio errors | A/V-check better by ≥ 5 points (H2) |
| Order consistency | ≥ 0.85 |
| Supported rate on correct control | 0.6-0.95, not saturated |

Probes, reported per backbone:

| Probe | Question |
|---|---|
| Vacuous sentences | Does the check return vacuous, not supported? |
| Fusion laundering | Does AV-pair reward new perceptual claims moved into `<fusion>`? |
| Question-text leakage into A-check | Does a claim taken from the question pass the A-check without audio support? |

A backbone failing Gate 0 is dropped from H1 and reported as such. Fewer than two pass → stop and report. If
Qwen3-Omni fails, the full ablation programme (pilot, tuning, Phase 3, H4) moves to the passing backbone with the highest isolated detection. A
pass on Gemini only means the idea needs an external critic, which is a different paper.

**Phase 2: pilot.** Before it, lock XRS and freeze XM-Test. Qwen3-Omni, A1 vs A8, 1,600 steps, one seed, dev
only: no benchmark and no XM-Test. Check: rewards are not constant, critic health stays out of saturation,
length stays bounded, and XM-Dev balanced accuracy moves. Estimate power from XM-Dev variance.

**Phase 2b: tuning.** §5.3.

**Phase 3: Full ablations** on Qwen3-Omni, 6,400 steps, seeds as in §7.4.

**Phase 4: Cross-backbone** on AVF and Qwen2.5-Omni: A0, A1, A2, A8, A15.

**Phase 5: Audit and write.** Human panel, audit judges, tables, paper.

## 10. Known traps

| # | Trap | Control |
|---|---|---|
| T1 | Critic drift to leniency (S_m→1) | Guard condition 2; supported-rate alarm; block-rate alarm |
| T2 | Collusion: actor learns phrases the critic likes | Offline post-run audit judges and reward–audit correlation |
| T3 | Reasoning grows, or shrinks to game S_m | max_new_tokens 512, hard cap; min(1, U_m/N) with N=3; length tracked |
| T4 | "No" bias from swapped data | 50% aligned controls; balanced accuracy; Yes% |
| T5 | Unimodal A-check cannot verify speech-lip or AV-sync claims | `<fusion>` may relate claims already made in the segments, scored by AV-pair, but adds no new perceptual claim |
| T6 | Label noise in swaps (a dog clip with barking in the background) | Label-disjoint pairs; SigLIP-verified visual labels; similarity tiers; human check of XM-Test |
| T7 | Backends differ in frames and audio windows | Fix per-model settings; compare within a backbone, never across |
| T8 | Silent modality failure (audio not decoded) | Input-assert tests; XM probes before each run |
| T9 | Benchmark solvable unimodally | Report A-only and V-only baselines beside AV |
| T10 | Reward gain is from format alone | A1 uses the same output format |
| T11 | Selection on test | §5.3; §5.4 T_max only; XM-Test frozen and hashed |
| T12 | AVF non-commercial licence, fork | Research only; pin the fork commit |
| T13 | Critic cost dominates the run | Log step time; budget tables §3.4, §6 |

## 11. Repository layout

```
docs/PLAN.md             this plan
avrsi/parse.py           segment and sentence parsing, dedupe, format reward
avrsi/rewards.py         R_outcome, R_pair, R_xm, group advantages
avrsi/critic.py          prompts, verdict parsing, critic views, budget planner
avrsi/sync.py            periodic sync and guard
avrsi/grpo.py            GRPO + per-token KL objective, no clipping
avrsi/perturb.py         swap planning, twin_id, planted errors
avrsi/metrics.py         §7 metrics
avrsi/stats.py           paired bootstrap, intersection-union, Holm, McNemar
avrsi/judges/            Gemini, OpenAI clients; cached; key handling
avrsi/data/              benchmark adapters
avrsi/models/            backend interface; Qwen3-Omni, Qwen2.5-Omni, AVF; fake backend
avrsi/train.py           training loop
scripts/                 Gate 0, eval, audit
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
