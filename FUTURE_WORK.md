# Future work

Roadmap of follow-up experiments and proposed innovations for the next
submission of *Benchmarking Faithfulness of Mathematical Chain-of-Thought
Autoformalisation*. Two sections:

1. **Additional experiments** — runs the rebuttal commits us to, plus the
   one item Luke explicitly green-lit on Discord (2026-04-24).
2. **Proposed innovations** — research directions beyond what was promised,
   each chosen because it directly answers a reviewer critique that the
   rebuttal could only patch verbally.

Priority ordering (impact / engineering cost) is given at the end.

---

## 1. Additional experiments

### 1.1 Re-run LLM perturbation with the BrokenMath prompt

**Motivation.** GPT-5.2-as-judge currently estimates the ineffective-perturbation
rate at **3.26%** for regex perturbation but **10.45%** for LLM-medium. The LLM
number is high enough that reviewers can use it to undercut the LLM-perturbation
arm of the comparison. Ighina flagged this on 2026-04-24; Luke replied
"yes please" the same day.

**Plan.**
- Replace the current LLM-perturber prompt with BrokenMath's exact prompt template.
- Re-run perturbation over the existing DAG corpus.
- Recompute the IPF rate (see §1.2) and the downstream TPR/FPR/FI on the new
  perturbed set.

**Code touchpoints.** `src/veriform/perturbation/perturbers.py` (LLM perturber),
`src/veriform/perturbation/brokenmath_perturber.py` (currently scaffold).

**Success criterion.** LLM-medium ineffective rate drops to a number we can
defend (target: comparable to regex). If it does not, that becomes a finding
in itself — argues that the LLM-perturbation arm is fundamentally noisier than
regex and should be deprecated or kept only as ablation.

---

### 1.2 IPF as a real component, with vs without ablation

**Motivation.** The rebuttal to Reviewer 4 describes IPF as a deterministic
typecheck-based filter and Appendix C.1 promises "results without applying
such pruning" alongside the pruned numbers. Today, IPF lives only in the
paper; the repo has `src/veriform/perturbation/ipf.py` as a scaffold.

**Plan.**
- Implement IPF: parse the perturbation from the perturbed NL → inject it
  into the unperturbed Lean → typecheck via the existing Lean server →
  classify perturbation as ineffective if the perturbed Lean still typechecks.
- Run the full benchmark in two modes: with IPF pruning and without.
- Report both in Appendix C.1 to make good on the rebuttal.

**Code touchpoints.** `src/veriform/perturbation/ipf.py` (fill in),
`src/veriform/proving/lean_server/` (typecheck call), `src/veriform/pipeline.py`
(filter step).

**Success criterion.** Pruned vs unpruned numbers differ by a small margin
(few percentage points); main story unchanged. This is the strongest available
defence against Reviewer 4's "perturbation validity assumption" critique
without invoking human annotation.

---

### 1.3 DAG-construction quality eval with a non-GPT model family

**Motivation.** Reviewer xHts: using GPT-5.2 to construct *and* evaluate DAGs
is circular. Rebuttal commits to redoing the Appendix B.2 quality eval with a
different model family.

**Plan.**
- Re-run the Appendix B.2 self-judge protocol with Gemini and/or Claude in the
  judge role over the same DAGs GPT-5.2 produced.
- Report agreement matrix and update Appendix B.2.

**Code touchpoints.** None inside the repo — DAG construction is upstream
(per CLAUDE.md, lives outside this repo). Just needs a one-shot batch eval
script against the saved DAG corpus.

**Success criterion.** Cross-judge agreement >= 80–85% on the random sample
(numbers TBD pending pilot).

---

### 1.4 Semantic-similarity re-weighting via the ProofBridge encoder

**Motivation.** Every reviewer raised some version of the "narrow notion of
faithfulness" critique. Rob's Discord proposal: re-weight TPR/FPR by an
embedding-similarity score between the NL statement and the formalised Lean,
using a contrastively-trained encoder à la ProofBridge. The ProofBridge team
has agreed to share their encoder; we use it out-of-the-box (Discord 2026-04-24,
Ighina: "for our next iteration we better keep focused on just using their
model out-of-the-box").

**Plan.**
- Drop the ProofBridge encoder behind `src/veriform/semantics/` (currently
  scaffold).
- Compute per-pair NL↔Lean cosine similarity over the existing benchmark output.
- Add a "semantic drift filter" mode: discard pairs whose similarity is below
  threshold τ, then recompute TPR/FPR/FI.
- Report both: raw FI and τ-filtered FI, swept over τ.

**Code touchpoints.** `src/veriform/semantics/` (encoder wrapper, similarity,
drift filter), `src/veriform/evaluation/` (re-weighted metrics).

**Success criterion.** Re-weighted FI preserves the headline finding
(sycophancy-vs-capability tradeoff) while quantifying semantic drift. Even a
modest correlation between similarity and human-judged faithfulness on a small
calibration set would address the "you don't measure semantic faithfulness"
line of attack head-on.

---

## 2. Proposed innovations

### 2.1 Reasoning-trace audit for direct sycophancy evidence

**Motivation.** Reviewers xHts and 2Ywr both argue that what we call sycophancy
is just a training-bias artefact. Stepfun and Goedel emit `<think>` traces
*before* the Lean. If those traces literally narrate self-correction
("the user wrote 5 but the correct answer is 6, I'll formalise 6"), that is
a smoking-gun rebuttal — the model is not mechanically substituting, it is
*deliberately* overriding the input.

**Plan.**
- The current pipeline already saves `formalizer_output` (raw model text) in
  every node pickle. Mine those strings.
- Build a small classifier (regex + LLM-judge) for self-correction language.
- Cross-tabulate: of the perturbed nodes that flip to "Proved" (i.e. likely
  sycophantic), what fraction contain explicit self-correction language in
  the trace?

**Code touchpoints.** New `scripts/analyze_traces.py` reading existing pickles
under `data/regex_perturbed/formalized/`. No new model runs needed.

**Why it's novel.** This is the only experimental setup that distinguishes
*latent training bias* (the model just produces the right answer because that
is its prior) from *active sycophancy* (the model recognises the discrepancy
and chooses to override). Reviewers cannot dismiss explicit traces as a
training artefact.

---

### 2.2 Sycophancy-vs-perturbation-magnitude curve

**Motivation.** xHts and 2Ywr complain the experimental section "describes
results without analysing them." A characterisation experiment turns a single
number into a story.

**Plan.**
- Define perturbation severity levels:
  - small: single-digit flip on one numeric literal
  - medium: sign or operator swap on one literal
  - large: chain-wide perturbation across multiple steps
- Run all four formalisers at each severity level over a fixed sub-sample.
- Plot FPR (sycophancy) and AF-fail rate as a function of severity.

**Predicted shape.** Sycophancy rate should be high at small severity (model
silently repairs) and drop at large severity (model gives up and emits
AF-fail). The diagnostic sweet spot is the middle.

**Code touchpoints.** Extend `StandardPerturber` with a severity knob;
`src/veriform/perturbation/perturbers.py`. New driver `scripts/run_severity_sweep.py`.

**Why it's novel.** Converts FaithformBench from a one-shot pass/fail metric
into a *characterisation tool* — researchers can locate where on the
severity curve their model degrades. Anticipates the "what is the
benchmark actually measuring" critique that we expect from the next round
of reviewers.

---

### 2.3 Faithfulness fine-tuning recipe — turn FaithformBench into a training set

**Motivation.** All reviewer rejections boil down to: "your benchmark detects
something, but is it a real failure mode or just a training artefact, and
what should we do about it?" The strongest possible answer is *we trained it
away*.

**Plan.**
- Construct (perturbed-NL, faithfully-perturbed-Lean) training pairs by
  injecting the same deterministic perturbation that produced the perturbed NL
  into the unperturbed Lean (this is exactly what IPF already does internally
  — same machinery, different output).
- SFT a small AF (e.g. Kimina-7B or a base Qwen-7B) on a mixture:
  - p% (correct-NL, correct-Lean) — preserves validity preservation
  - (1-p)% (perturbed-NL, faithfully-perturbed-Lean) — teaches faithful
    translation of incorrect inputs
- Evaluate the fine-tuned model on FaithformBench.

**Success criterion.** Lower FPR (less sycophancy) with no significant TPR
drop. This would make the paper *prescriptive*, not just diagnostic, and
elevates the contribution from "we found a problem" to "we found a problem
and a fix."

**Risks / open questions.**
- Need to confirm the synthetic faithfully-perturbed Lean is itself
  syntactically and semantically clean — IPF's typecheck gives us syntactic
  guarantees but not full semantic ones.
- Catastrophic forgetting on standard AF benchmarks (miniF2F etc.); need a
  held-out validity check.
- Compute: SFT on a 7B AF for ~1 epoch on a few thousand examples is
  feasible on a single H200 over a couple of days.

**Code touchpoints.** New `scripts/build_faithful_sft_data.py`, then standard
HF / TRL fine-tuning. Training infra is not currently in the repo.

**Why it's novel.** No prior work has used a faithfulness-perturbation
benchmark *as its own training signal* for the AF model. This closes the loop
and makes the benchmark useful beyond evaluation.

---

### 2.4 Cross-AF agreement as a cheap sycophancy signal

**Motivation.** DSP-V2 is the dominant runtime cost of the benchmark. A
proxy that doesn't need a prover would let the methodology scale to
much larger corpora.

**Plan.**
- For a perturbed NL input, run multiple AFs (we already have stepfun, kimina,
  goedel; herald soon) and compare their Lean outputs.
- Define a Lean-statement equivalence check (existing theorem extractor +
  normalised string match, or BEq if available).
- Hypothesis: when two or more AFs converge on the *same* Lean from a
  perturbed input, this is strong evidence that both silently auto-corrected
  to the same canonical form — i.e. sycophancy.
- Validate by correlating cross-AF agreement against the DSP-V2-derived
  sycophancy label on the existing benchmark output.

**Code touchpoints.** `src/veriform/proving/theorem_extractor.py` (already
extracts statements), new `scripts/cross_af_agreement.py` reading existing
pickles. No new model runs if all four AFs have already been formalised.

**Why it's novel.** Proves that sycophancy can be detected without a prover,
which (a) cuts the cost of FaithformBench by an order of magnitude and
(b) opens the methodology up to domains where a prover doesn't exist
(e.g. SQL, program synthesis — the generalisation question Reviewer kMrF
raised).

---

## Priority ordering

By impact / engineering cost, in execution order:

1. **§1.1** BrokenMath-prompt LLM rerun — highest defence value per hour of work; explicitly green-lit.
2. **§1.2** IPF implementation + ablation — promised in rebuttal; required for §1.1's IPF-rate number.
3. **§2.1** Reasoning-trace audit — nearly free given saved artefacts; addresses the most damaging reviewer critique (training bias vs sycophancy).
4. **§2.3** Faithfulness fine-tuning — biggest paper upside (turns paper prescriptive); largest engineering cost.
5. **§1.4** Semantic re-weighting — blocked on ProofBridge encoder delivery from their team.
6. **§2.2** Severity curve — straightforward sweep, good characterisation.
7. **§1.3** DAG cross-judge eval — promised, cheap, low-impact.
8. **§2.4** Cross-AF agreement — interesting but the existing prover-based pipeline already works; this is an optimisation.
