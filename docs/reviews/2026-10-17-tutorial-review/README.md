# 2026-10-17 tutorial review: record and caveats

Files here are evidence, kept as recorded. Read these caveats before quoting them.

- `scgpt_single_cell_colab_Review.md`, `findings.jsonl`: the review (findings T80-1..T80-6) at base `4875d9b`.
- `t80_1_repro.py` -> `t80_1_results.json`: the reviewer's CPU reproduction. Its masks and binning match this API's: on the
  32 synthetic cells `t80_1_verify.py` reproduces it through the API exactly (`t80_1_verify.json`). Protocol labels for the
  real-PBMC Pearsons: MLM call 0.314; generative call 0.371 with `<cls>` value -2 (this API's default) or 0.390 with `<cls>`
  value 0; MVC head 0.551 with `<cls>` value -2.
- `t80_1_lineage.py` -> `t80_1_lineage.json`: **historical, different protocol.** Its generative queries remove the queried
  genes before re-binning, include undetected markers, and use `<cls>` value 0; the API bins the full cell, masks only
  detected retained markers, and defaults to -2. Its lineage differences therefore do not validate the API's lineage
  behaviour, and the notebook and docs no longer quote them.
- `t80_r2_pbmc.py` -> `t80_r2_pbmc.json` (round 2, Codex peer review): the same 32 pbmc3k cells through **this API**, with a
  per-gene prior (mean bin over 200 other pbmc3k cells), a constant baseline (mean bin of the cell's unmasked genes) and a
  shuffled-context control, over mask seeds 42-46. Seed 42 reproduces 0.314 / 0.3712 / 0.551; the gene prior
  scored 0.5756, and the MVC head with another cell's context 0.4782. These are reconstruction associations,
  not proof of contextual recovery.
- `run3_probe.log`: the notebook's probe stage on the default sample (CPU), including per-cell values and 5-seed summaries.
- All numbers are CPU (torch 2.14.1+cpu) and descriptive; masked genes within a cell are not independent replicates.
