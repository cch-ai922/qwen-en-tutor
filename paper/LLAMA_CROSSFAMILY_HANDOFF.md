# Cross-family (Llama-3.2-1B) replication — handoff

Prep for the §6.3 cross-family experiment: a **trained** non-Qwen student, the
one substantive item left from the reviewer's list. Everything below is
**prepared and validated**; nothing has been trained yet. Start when the GPU is
free (targeted ~12h out).

## TL;DR — one command

```powershell
cd qwen-en-tutor
.\.venv\Scripts\Activate.ps1
pwsh -File scripts\run_llama_crossfamily.ps1
```

Then run the judged-withholding step it prints at the end (needs a judge
server; see §4 below). Resumable — re-run verbatim if interrupted.

## What was prepared

| Asset | Path | Purpose |
|---|---|---|
| A1 config (full mix) | `config/paper/training_llama_a1_full.yaml` | SFT-only, full 12-stream corpus |
| A3 config (generic-SFT) | `config/paper/training_llama_a3_no_specialized.yaml` | SFT-only, no specialized streams — the baseline |
| Eval baselines | `scripts/run_paper_eval.py` → `paper_llama_a1_sft`, `paper_llama_a3_sft` | drive the frozen eval sets |
| Runbook | `scripts/run_llama_crossfamily.ps1` | train → generate → mechanical-score |

## Design decisions (and why)

1. **SFT-only, A1 + A3 only.** These two conditions reproduce the *load-bearing*
   contrast the boundary rests on (full mix vs generic-SFT). We do NOT replicate
   A5/decorrelation cross-family — that side-result is Qwen-scoped by design.
2. **Reuse the existing corpus.** `data/sft_filtered/` and `data/sft_filtered_a3/`
   are teacher-distilled and family-agnostic on the tutor side. No regeneration.
3. **Base = Llama-3.2-1B-*Base*.** Clean base-vs-base match to Qwen A1 (also
   trained from a base checkpoint) — no train-from-instruct asymmetry to caveat.
4. **Identical recipe** (r=16, α=32, 2 epochs, lr 2e-4, eff. batch 8, seed 42,
   same 7 target modules) so A1-vs-A3 stays single-variable within the family.

## Integration points — patched and verified

Two code changes were needed for the Base checkpoint (both verified against the
vendored model + a real SFT record before any training):

- **SFT formatter is now family-aware** (`src/qwen_tutor/training/formatter.py`).
  It was hardcoded to Qwen ChatML markers (`<|im_start|>`/`<|im_end|>`) in three
  places: the special-token validator (your original crash), the assistant-span
  marker scan used for loss masking, and the `/no_think` injection. It now
  detects the family from the tokenizer vocab and uses Llama-3 markers
  (`<|start_header_id|>assistant<|end_header_id|>\n\n` … `<|eot_id|>`), skips the
  Qwen `/no_think` tag for Llama, and validates the right markers. **Verified:**
  Llama masks 1366/1570 tokens with only assistant turns labeled; Qwen regression
  unchanged (201 labeled). Correct masking confirmed — no silent bad training.
- **Eval generate stops on all turn-end markers**
  (`src/qwen_tutor/training/eval/run_eval.py`, `HFTargetModelClient._generate_sync`).
  This is the path paper numbers actually go through:
  `run_paper_eval.py` → `HFBaseline` → `HFTargetModelClient`. It already halted on
  `<|im_end|>`/`<|endoftext|>` but NOT `<|eot_id|>`. Llama-Base declares
  `eos=<|end_of_text|>` (128001) while the template ends turns with `<|eot_id|>`
  (128009), so without adding eot a trained Llama student runs to `max_new_tokens`
  every turn. Now halts on the eos plus any of `<|eot_id|>`/`<|im_end|>`/`<|endoftext|>`
  present in the vocab. Verified stop set: Llama `[128001, 128009]`, Qwen
  `[248044, 248046]`. Foreign markers resolve to `None` and are filtered — no
  false stop, and Qwen behavior is unchanged.

Verified-safe without changes:

- **Arch routing:** `model_loader.py` reads `architectures=["LlamaForCausalLM"]`,
  loads via `AutoModelForCausalLM`.
- **pad token:** Llama-Base has none; `sft.py` sets `pad_token = eos_token` at load.
- **Chat template:** supports a `system` role; the `enable_thinking` kwarg is
  try/excepted, so Llama's template (which lacks it) falls back cleanly.

## 1. Train (Stage 1 of the runbook)

Two SFT runs, ~overnight each on the 3060 (mirror the Qwen A1 ~3–5h/epoch × 2).
Adapters land in `outputs/paper/llama_a1/sft` and `outputs/paper/llama_a3/sft`.
**Stop the teacher server first** — training + teacher don't co-fit in 12GB.

## 2. Generate on frozen eval sets (Stage 2)

`run_paper_eval.py --baseline paper_llama_a{1,3}_sft --test-set all`. Local hf,
no teacher. Writes `outputs/paper/eval/paper_llama_a{1,3}_sft/<set>.jsonl`.

## 3. Mechanical scoring (Stage 3) — persistence + locale, judge-free

Produces persistence sentinel recall (the single cleanest boundary metric) and
locale leakage. In `outputs/paper/score/mechanical/paper_llama_a{1,3}_sft/`.

## 4. Judged withholding (manual — needs a judge server)

Not auto-run (the judge GGUF competes for the GPU). Start ONE judge on :8080,
then per judge:

```powershell
python scripts/score_withholding_rate.py --judge llama31_8b_judge `
    --eval-dir outputs/paper/eval `
    --baselines paper_llama_a1_sft,paper_llama_a3_sft `
    --out outputs/paper/score/llama_withholding_llama31.json
# swap to gemma2_9b_judge, repeat, then average the two (paper's 2-judge protocol)
```

## 5. Reporting (what to put in the paper)

Compare the Llama A1/A3 numbers to the Qwen rows. The claim to test is
**direction**, not magnitude parity:
- persistence recall: Llama-A1 (trained) ≫ Llama-A3 (generic-SFT) — does SFT
  install the sentinel behavior in a second family?
- withholding: Llama-A1 > Llama-A3, and ideally > prompt-only Llama-3.1-8B
  (already in the paper at §6.2).

If direction holds, flip the §6.3 item-4 "concrete plan" text and the abstract's
"single family" scoping to "demonstrated in a second trained family," and add a
Llama row to the Table 9b statistical summary. If it does NOT hold, that is
itself a reportable, honest result about family-dependence — do not bury it.

## Known risks to watch on the actual run

- **First-epoch loss sanity:** if SFT loss is ~0 or constant, the system prompt
  is eating all labels (max_seq_length too small for the tokenizer's rendering).
  Llama tokenizes the prompt a bit differently than Qwen; if this bites, raise
  `max_seq_length` 1792 → 2048 (a 1B model in 4-bit has the headroom).
- **Chat-template SFT masking:** confirm the trainer masks the prompt and trains
  only on assistant turns (same as Qwen). If withholding/persistence come out
  near-random, check the label mask first.
