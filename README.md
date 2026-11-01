# Gemma 4 Context Pruning

Kaggle notebook: https://www.kaggle.com/code/aks1321/gemma4-context-pruning

Project page: https://arjunkshah12345-hash.github.io/gemma4-context-pruning/

## Files
- `gemma4_context_pruning.ipynb`: the full harness, pruner, benchmark and evaluation (runs in DRY mode without a GPU).
- `results/`: GPU run outputs (`results.json`, `summary.csv`, run log) and CPU pruning-only outputs (`results_cpu.json`, `summary_cpu.csv`).

### Query-Aware Context Pruning for Small Offline Coding Agents: How Much Repository Context Does Gemma 4 Actually Need?

**Track:** Code Comprehension / Tasks and Benchmarks
**Author:** Arjun Shah

> STATUS: the harness, pruner, benchmark and pruning-only measurements below were run and verified on CPU. Section 5.2 and the model sentences come from the Gemma 4 E4B GPU run of `gemma4_context_pruning.ipynb` on Kaggle (MAX_ATTEMPTS=1, greedy, bf16); token fractions there use the Gemma tokenizer.

### 1. Abstract
Small open models running offline have tight context windows and slow prefill, yet coding agents routinely paste whole files, stack traces and test logs into the prompt. We study a cheap, query-aware line filter that keeps lines lexically and structurally related to the failing test, and measure its effect on a Gemma 4 agent. On a 12-task Python bug-fix benchmark, the filter keeps the buggy function intact in 12 of 12 tasks at every budget down to 15% of lines, while random pruning at 30% keeps it in 0 of 12. On a Kaggle GPU, Gemma 4 E4B fixes 12 of 12 tasks at every query-aware budget down to 15% of lines (27% of Gemma tokens), with mean prefill time falling from 14.35 s to 4.58 s (3.1x faster). Controls show that accuracy is not a discriminating signal on this set: random 30% pruning still solves 11 of 12, pruning without anchors solves 10 of 12, and a traceback-only prompt (about 1.1% of tokens) solves 12 of 12, so these tasks are largely solvable from the issue text and traceback. The demonstrated benefit is lower prefill latency and cost; buggy-function retention is the structural evidence that the filter keeps the right code. We release the filter, the evaluation harness and per-task traces.

### 2. Motivation
The main competition asks for an offline agent that fixes real Python bugs under fixed compute. Prefill dominates latency for long prompts, and irrelevant context can also distract a small model. A pruning step that runs in milliseconds on CPU is a free lever if it does not hurt accuracy. The open question is where the accuracy cliff sits.

### 3. Method
**Agent loop.** Gemma 4 receives the issue text, the failing test, its traceback and a context bundle. It replies with the corrected definition of the target function; the harness locates that function with Python's `ast`, splices it into the file and reruns the test. Replies that do not parse or do not contain the function count as apply failures. One attempt per task by default (`MAX_ATTEMPTS`).
**Context bundle (baseline).** The full target module.
**Pruner.** Each line gets a score from (a) identifier overlap with the issue, test and traceback, (b) proximity to an anchor line, (c) a large bonus for structural lines (`def`, `class`, `import`, decorators). The enclosing block of every anchor is always kept, then the top-scoring remaining lines fill the k% line budget. Anchors are traceback frames inside the target file; when there are none, anchors fall back to the `def` lines of functions named in the traceback.
**Budgets.** k in {100, 75, 50, 30, 15}% of lines.
**Ablations (at 30%).** Drop identifier overlap; drop the structural bonus; drop anchors entirely; random pruning; traceback-only (anchor blocks alone).

### 4. Experimental setup
- Model: Gemma 4 E4B instruction-tuned (`google/gemma-4/transformers/gemma-4-e4b-it`), Kaggle GPU, greedy decoding, seed 0, thinking disabled.
- Tasks: 12 single-function Python bugs (off-by-one, wrong operator, lost ordering, wrong even-length median, and similar), each placed in the middle of a module padded with 40 unrelated helper functions (mean module size about 1,948 tokens under the CPU proxy tokenizer). Every task fails before fixing, and a reference fix for each passes through the same apply path the model uses (verified 12/12). The notebook also accepts external tasks via `TASKS_JSON`.
- Metrics: pass@1, tokens kept, prefill time (a separate forward pass over the prompt), wall-clock per task, apply failures, plus two model-free checks: whether the full buggy function survives pruning, and whether the pruned context still parses.

#### 5. Results

#### 5.1 Pruning-only measurements (measured, CPU, no model)
Token counts here use a word/punctuation proxy tokenizer; the GPU run recomputes them with the Gemma tokenizer.

| Variant | Tokens kept (mean fraction) | Buggy function intact | Context parses |
|---|---|---|---|
| 100% | 1.000 | 12/12 | 12/12 |
| 75% | 0.777 | 12/12 | 0/12 |
| 50% | 0.558 | 12/12 | 0/12 |
| 30% | 0.382 | 12/12 | 0/12 |
| 15% | 0.239 | 12/12 | 0/12 |
| Random 30% | 0.302 | 0/12 | 0/12 |
| No identifier term, 30% | 0.374 | 12/12 | 0/12 |
| No structural term, 30% | 0.379 | 12/12 | 0/12 |
| No anchors, 30% | 0.379 | 1/12 | 0/12 |
| Traceback-only | 0.012 | 12/12 | 12/12 |

#### 5.2 Model results (Gemma 4 E4B)
| Variant | Tokens kept | pass@1 | Prefill time | Apply failures |
|---|---|---|---|---|
| 100% | 1.000 | 12/12 | 14.35 s | 0 |
| 75% | 0.793 | 12/12 | 11.40 s | 0 |
| 50% | 0.586 | 12/12 | 8.65 s | 0 |
| 30% | 0.419 | 12/12 | 6.49 s | 0 |
| 15% | 0.270 | 12/12 | 4.58 s | 0 |
| Random 30% | 0.298 | 11/12 | 4.97 s | 0 |
| No identifier term, 30% | 0.398 | 12/12 | 6.24 s | 0 |
| No structural term, 30% | 0.415 | 12/12 | 6.47 s | 0 |
| No anchors, 30% | 0.416 | 10/12 | 6.49 s | 0 |
| Traceback-only | 0.011 | 12/12 | 1.35 s | 0 |

### 6. Analysis
- **Prefill latency is the demonstrated win.** Query-aware pruning keeps pass@1 at 12/12 at every budget while mean prefill time falls from 14.35 s (100%) to 11.40 s, 8.65 s, 6.49 s and 4.58 s at 75%, 50%, 30% and 15% of lines, a 3.1x reduction at 15%. Mean wall-clock per task falls from 32.10 s to 13.85 s over the same range.
- **Accuracy preservation is not discriminating on this set.** Random 30% pruning, which never keeps the full buggy function, still scores 11/12. Pruning without anchors scores 10/12. Traceback-only, at about 1.1% of tokens, scores 12/12 with 1.35 s prefill. Gemma 4 E4B can usually write the fix from the issue text, failing test and traceback alone, so 12/12 at 15% shows that pruning does not hurt on these tasks, not that the retained code is what made them solvable. We did not observe pruning raising pass@1, because the full-context baseline is already at ceiling.
- **Buggy-function retention is the structural evidence.** With the Gemma tokenizer, the buggy function survives intact in every task (retention 1.0) at every query-aware budget, versus 0.0 for random 30% and 0.083 (1 of 12) without anchors. The two ablations with lower retention are also the only two that lose tasks, which is consistent with, but on 12 tasks does not establish, retention mattering for accuracy.
- **Anchors carry the pruner.** Removing either the identifier or the structural term leaves retention at 1.0 and pass@1 at 12/12; removing anchors drops both.
- **Raw tracebacks rarely point into the buggy file.** For all 12 tasks the failing assertion is raised in the test file, so no traceback frame lands in the target module. The name-based fallback (functions named in the traceback) was needed in every case.
- **Line-level pruning breaks syntax without hurting the model here.** Every pruned bundle below 100% fails to parse, yet there were 0 apply failures in any variant and no pass@1 loss for query-aware budgets.
- **Token fraction exceeds line fraction** (0.419 of Gemma tokens at 30% of lines) because the always-kept signature lines are longer than average.

Harder tasks where the fix cannot be inferred from the traceback, such as multi-file bugs or bugs whose cause lies away from the failing call, are needed to test whether query-aware selection preserves accuracy where random or traceback-only context does not. We leave these to future work.

### 7. Limitations
Lexical scoring misses semantic dependencies (dynamic dispatch, config-driven code). The built-in tasks are small single-function bugs with synthetic padding, so they test retrieval of the right function more than multi-file reasoning. Results are for one model family and one task distribution.

### 8. Reproducibility
One notebook reproduces everything. On a GPU session it loads Gemma 4 through `kagglehub` and writes `results.json` (per-variant, per-task rows including truncated model replies) and `summary.csv`. Without a GPU it runs in DRY mode and reproduces Table 5.1 exactly.

### 9. Conclusion
A millisecond, CPU-only pruner anchored to functions named in the failure keeps the buggy function intact at every budget down to 15% of lines. On Gemma 4 E4B this cuts mean prefill time 3.1x (14.35 s to 4.58 s) with pass@1 unchanged at 12 of 12. Because random pruning (11/12), no-anchor pruning (10/12) and traceback-only prompts (12/12) also score highly, this benchmark shows that pruning is safe and cheaper, not that it preserves accuracy where less targeted context would fail. Testing that claim requires harder, multi-file tasks.

## License
MIT
