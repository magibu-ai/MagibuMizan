# Evaluation notes

These are results from the September 2026 Magibu experiments. The open models were **not fine-tuned** for this evaluation. Each question's option labels were read once in normal order and once in reverse order; the aligned distributions were averaged. A read means one prompt evaluation ending at the option-label token, with no answer text generated.

The MMLU experiments used a separate leaderboard-prompt evaluation harness in `letter_readout`, **not the public MagibuMizan API prompt**. The tool-selection experiment used full tool-name likelihoods rather than A–Z labels, and its scorer is not part of this repository. These measurements describe the named experimental setups; they are not guaranteed performance for an arbitrary MagibuMizan deployment.

## Accuracy

The numbers before and after `/` are the normal-order read and the two-order average. Jev results are from public traces; we did not run the Jev model ourselves.

| Task | Gemma 4 26B-A4B | Qwen3.5 27B | Qwen3.8 27B | Jev 1.13.0 |
|---|---:|---:|---:|---:|
| Turkish MMLU, 6,200 questions, 5 options | 76.6% / 77.7% | 78.6% / 79.4% | 77.1% / 78.4% | 75.9% |
| Turkish MMLU-Pro preview, 12,000 questions, 10 options | 58.0% / 59.5% | 58.6% / 61.0% | 58.9% / 60.6% | 59.6% |

The open models ran in bf16 on one NVIDIA H200. Approximate H200 time per read on MMLU was 13 ms for Gemma, 37 ms for Qwen3.5, and 53 ms for Qwen3.8. Jev API latency is not directly comparable because its questions were sent in batches. MMLU-Pro was an unfinished preview with incomplete subject coverage and potentially weak or duplicate added options; its results are exploratory.

## Calibration

We fitted one temperature per model by minimizing negative log-likelihood on alternating questions, then measured the other half. ECE is 15-bin expected calibration error on the held-out half. Coverage is the share of held-out questions that can be accepted in descending confidence order while the observed error rate stays at or below 5%. These values depend on the task, prompt, model checkpoint, and option count.

| Measure, two-order average | Gemma 4 26B-A4B | Qwen3.5 27B | Qwen3.8 27B | Jev 1.13.0 |
|---|---:|---:|---:|---:|
| MMLU ECE, raw → scaled | 0.121 → 0.025 | 0.049 → 0.018 | 0.017 → 0.023 | 0.035 → 0.027 |
| MMLU-Pro ECE, raw → scaled | 0.239 → 0.043 | 0.121 → 0.045 | 0.083 → 0.033 | 0.044 → 0.035 |
| MMLU coverage at ≤5% observed error | 57% | 63% | 61% | 54% |
| MMLU-Pro coverage at ≤5% observed error | 25% | 32% | 30% | 27% |

The 4-bit `mlx-community/gemma-4-26B-A4B-it-qat-4bit` checkpoint was checked separately on 620 MMLU questions (10 per subject) on an M2 Pro: 73.4% in normal order and 75.2% after order averaging, versus 76.8% / 78.1% for bf16 on the same questions. The MMLU-fitted temperatures were **2.5 for this 4-bit checkpoint** and **4.35 for bf16 Gemma**. They are examples of task-specific calibration, not defaults for every application.

## Separate tool-selection experiment

The public [Jev Turkish tool-call traces](https://huggingface.co/datasets/aliarda/jev_turkish_tool_call_traces) contain 308 turns, including 83 labelled `no_tool`, with 247 candidate tools. The experiment scored each full tool name and end-of-turn probability; it did not call `MagibuMizan.answer` or use letter labels. Correct counts for one option order were Gemma **247**, Qwen3.5 **241**, Qwen3.8 **246**, and Jev **246**. At a 5% observed-error budget, coverage was 17%, 22%, 26%, and 52%, respectively. The `no_tool` convention accounts for many errors, so these figures should not be generalized to other tool-routing datasets.

## Sources and reproduction boundary

- [Turkish MMLU](https://huggingface.co/datasets/alibayram/turkish_mmlu) and [Turkish MMLU-Pro preview](https://huggingface.co/datasets/alibayram/turkish-mmlu-pro-preview) provide the question sets. The MMLU run used the leaderboard's Firebase question snapshot and prompt; Pro used the preview dataset. Those datasets have their own access and license conditions; their question text is not redistributed here.
- [Jev Turkish MMLU traces](https://huggingface.co/datasets/aliarda/jev_turkish_mmlu_traces) and the tool-call traces above supplied Jev outputs. The listed Jev scores are observations from those traces.
- The table is an aggregate of local experiment outputs. Raw prediction files and the separate evaluation harness are not shipped in this small runtime repository, so the tables are documented observations rather than an end-to-end reproducible benchmark bundle.
