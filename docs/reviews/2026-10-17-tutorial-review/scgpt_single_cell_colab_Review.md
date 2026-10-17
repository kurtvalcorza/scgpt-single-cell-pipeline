# Row 80 review: scgpt-single-cell-pipeline, `tutorials/scgpt_single_cell_colab.ipynb`

- Target: `kurtvalcorza/scgpt-single-cell-pipeline` main `4875d9bf83e767745976f1aced3657b082fcc8ea` (re-resolved 2026-10-17; main has not moved). Notebook blob `07755aa2ddf8c6a1fee7ea38964276b59b41f99d`, verified with `git hash-object --no-filters`.
- Spec baseline: NOTEBOOK_SPEC 2.5 (ml-worker main `bfe4150cad9d0e14ccd9905493e27383fa073a02`). The notebook declares 2.2.
- Hosted evidence for this exact blob: Colab CLI T4 run at `8d35011`, 2026-10-10, PASS, `restarted: false` (`docs/execution-evidence/2026-10-10-8d35011/`). REL14 holds. The registry says **Candidate**.
- Mode: read-only. Nothing was changed in the repository.
- **Readiness: Needs revision.** 2 Major, 4 Minor, 0 Blocker. 1 owner decision.

## T80-1 verdict: upstream behaviour plus the wrong attribution. There is no pipeline bug.

The suspected bug does not reproduce. The pipeline's masked-expression call is correct against upstream's MLM convention. The near-zero 0.0045 comes from the notebook's synthetic cells, not from a broken probe or a dead decoder.

**Checked against upstream** (bowang-lab/scGPT `cebd6fa`, plus the census pretraining script at `4068d67`):

| Item | Upstream | Pipeline | Match |
|---|---|---|---|
| MASK_VALUE / pad | DataCollator `mask_value=-1`; tutorials use `mask_value=-1` and `pad_value=-2` | -1; pads are excluded by the key-padding mask | yes (MLM convention) |
| Binning | normalise to 1e4, log1p, `np.quantile(nonzero, linspace(0,1,n_bins-1))`, `_digitize` spreads ties at random, n_bins 51 | the same, with a seeded tie spread | yes |
| Vocab, `<cls>` | `default_census_vocab.json` (60,694 genes) | TDC vocab: all 60,694 gene ids identical, and `<pad>`/`<cls>`/`<eoc>` appended | yes |
| `<cls>` value | 0 in the fine-tuning tutorials; `pad_value` -2 in pretraining | 0 | yes (fine-tuning convention); -2 makes almost no difference (see below) |
| Architecture | ContinuousValueEncoder (clamp 512, ReLU), post-norm layers, ExprDecoder (3 linears, LeakyReLU); `mlm_output = decoder(h)['pred']` | the same; key `pred` | yes |
| include_zero_gene | False in the annotation tutorial, True in the integration tutorial | zeros are never tokenised (equivalent to False) | yes |
| Decoder weights | the original `.bin` holds `decoder`, `mvc_decoder` (inner product), `flag_encoder` and a 177-class `cls_decoder` | the safetensors file keeps `expr_decoder` (bit-equal to the `.bin`) and drops `mvc_decoder` and `flag_encoder` | the decoder ships; GEP/MVC does not |
| dtype | fp16 autocast during pretraining | float32 | fine |

**How the checkpoint was pretrained.** In `examples/pretrain.py`, `--training-tasks` defaults to `both`, which turns on `USE_GENERATIVE_TRAINING`. Under it, the genes to be predicted enter as **gene embedding + `flag_encoder(1)`, with no value input**. Each of them attends to the known genes and to itself. The loss covers only those generated positions, plus MVC. The `flag_encoder` in the checkpoint corroborates this. So the -1 MLM probe is upstream's *fine-tuning* convention, not this checkpoint's *pretraining* convention. Cell 23 calls -1 "the training-time mask value", which is wrong.

**Reproduction on CPU** (torch 2.14.1+cpu, float32). Every method ran on the same cells with the same masks. The independent MLM re-implementation equals the pipeline's call to 4 decimals.

| Cells | Pipeline `predict_masked` (MLM -1) | Upstream generative (`<cls>` 0 / -2) | GEP/MVC head from `<cls>` |
|---|---|---|---|
| Real pbmc3k, 32 cells, 15 % masked | **0.314** | **0.390** / 0.371 | **0.551** |
| Real pbmc3k, 32 cells, 40 % masked | 0.340 | 0.411 / 0.392 | 0.553 |
| Notebook synthetic cells, 32 | 0.087 | 0.081 / 0.065 | 0.151 |
| Hosted run (8 synthetic validation cells) | 0.0045 | not run | not run |

- Under every convention, the expression decoder predicts bins in a narrow band of about 26–33. On pbmc its MAE is 11.9 bins against 11.7 for always predicting the mean bin. It is not an imputer.
- **Lineage, real PBMC** (12 T and 12 B cells; marker genes queried):
  - The MVC head predicts T markers at bin 35.2 in T cells and 31.3 in B cells, and B markers at 37.3 in B cells and 30.4 in T cells. The lineage conditioning is clear.
  - The generative expression decoder separates by +1.0 and +1.2 bins.
  - The pipeline's MLM probe separates by +0.7 and -0.2 bins.

**Verdict.** Upstream behaviour, wrongly attributed:

1. The synthetic cells have no co-expression structure, so every probe convention sits near chance on them.
2. On real cells, the same pipeline call gives r≈0.31. The checkpoint's own generative convention lifts that to ≈0.39, and the MVC head that TDC dropped reaches ≈0.55.
3. The expression decoder is genuinely compressed, so the profile steer ("embeddings and classification, not imputation") stands.
4. Two statements do not stand: "chance-level / a property of the shipped decoder" and "the decoder does not use lineage context".

**How to present it.** Reword cells 0, 23 and 25, objective 3 and the model card (owner decision T80-1; recommended pick **A**). If the owner picks B or C, real-PBMC Pearson rises from 0.31 to 0.39 (generative) or to 0.55 (MVC). Synthetic cells stay at 0.06–0.15.

## Findings

| ID | Sev | Title | Acceptance at SHA |
|---|---|---|---|
| T80-1 | Major | The masked-expression "chance" result is blamed on the decoder; the real causes are the synthetic data and the probe convention. **Owner decision** | fail |
| T80-2 | Major | DAT9 MUST: the notebook never mentions overlap with the CELLxGENE Census 2023-05-08 corpus, which includes normal blood cells, for the cited public-PBMC evidence or BYOD | fail |
| T80-3 | Minor | REL13: the notebook says "Kaggle T4 run" (it was Colab CLI), mentions a 504 that did not happen in this run, quotes 7 s (recorded 9.12 s) and bins 28–32 (recorded 27.04–32.02), and gives an unsourced p≈0.16 | fail |
| T80-4 | Minor | Declares spec 2.2 (current 2.5); no `notebook_kind`; no DOC17 use notice | fail |
| T80-5 | Minor | SEM-ENFORCE: the embedding checks and Section 10 parity print FAIL but do not raise. Reload parity does raise. | fail |
| T80-6 | Minor | DAT20: a BYOD refusal shows a stage traceback plus a kernel RuntimeError, and stale outputs stay on disk | fail |

Each finding's details are in `findings.jsonl`. `probes/acceptance_checks.py` fails all six at `4875d9b`.

## Round pattern checklist

| Check | Result | Finding |
|---|---|---|
| REL13 | fails | T80-3 |
| REL14 | holds: the exact blob has a recorded PASS run, and the registry says Candidate | — |
| Spec version | 2.2 → 2.5, owner migrates each cycle | T80-4 |
| DAT20 | deviates (SHOULD) | T80-6 |
| Recipes chosen on test | none found. Hyperparameters are fixed tutorial values; the final epoch is kept; validation is used only for monitoring; the test split is touched only in Section 10. | — |
| Held-out claims | appropriately hedged: n=16, synthetic, "saturated", no gain claimed | — |

Prior-round fixes (SCG-M1..M2, SCG-m1..m2, commit `56078fb`) hold. The one-pass run without a restart exists. There is a stage-import test.

## Probes

- `run_probes.py`, static: 10 pass, 1 fail (SEM-ENFORCE, verified by hand), 5 info, 7 not_verified (the execute probes did not run).
- `--evidence-nb` was the Colab CLI output notebook for blob `07755aa2`.
- SEM-NUMBERS passes, but only because it matches metric values. The REL13 mismatches in T80-3 were found by hand.

## Most likely wrong

The claim that whole-human was trained with the generative convention. It rests on the default `--training-tasks both` in the public census script plus the `flag_encoder` in the checkpoint. The exact command line used for the released checkpoint is not published.

## Files

- `findings.jsonl`, `return.json`
- `scripts/t80_1_repro.py`, `scripts/t80_1_lineage.py`
- `work/t80_1_results.json`, `work/t80_1_lineage.json`
- `probes/` (run_probes.py, acceptance_checks.py, results.json, acceptance_results.json, source_manifest.json)
- `row80_probes.zip`
