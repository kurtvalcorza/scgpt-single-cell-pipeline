# Weight provenance, the vocabulary's second source, and DIMER hosting

This repository pins **one** snapshot with its own `dimer-base-manifest.json`. One of its four entries does not come from the Hugging Face Hub.

## scGPT (whole-human, TDC safetensors) weights

- Upstream: `tdc/scGPT`
- Immutable revision: `acf749f35bf5c0b00633838f02588272ed0d9911`
- Weight format: SafeTensors (`model.safetensors`, 203,233,980 bytes, 159 tensors, 50,804,225 parameters)
- Upstream weight license: MIT (`license: mit` in the pinned upstream README front matter and in the Hub repository metadata; the original `bowang-lab/scGPT` release is MIT)
- Local layout: `weights/scgpt/` holds the 4 manifest entries (`config.json`, `model.safetensors`, `vocab.json`, upstream `README.md`; 204,555,393 bytes total) with byte size and SHA-256 for each. `verify_snapshot()` in `src/scgpt_single_cell_pipeline/pipeline.py` checks all of them before any load and refuses a manifest that omits the config, the weights or the vocabulary; `stage_missing_files(allow_download=True)` fetches only absent entries. `.safetensors` files are git-ignored; `vocab.json` is committed.
- Cross-check: the manifest's `model.safetensors` digest `cabd40e6e22514865825975940f2c61ac1395f3317bba9f773858cd72914064c` equals the `oid sha256` of the Hub LFS pointer at the pinned revision (`https://huggingface.co/tdc/scGPT/raw/acf749f35bf5c0b00633838f02588272ed0d9911/model.safetensors`).

## What the packaged checkpoint is, checked against the original

The Hub repository also carries `scgpt_gh_repo_original_model.bin` (207,861,754 bytes), the checkpoint the packager started from. It was loaded once at build time with `torch.load(weights_only=True)` in a scratch environment — never by the pipeline — and compared tensor by tensor with the safetensors file:

- 173 tensors in the original; 159 in the safetensors file. Every one of the 159 is **bit-identical** to its original after the packager's renaming (`encoder.*` → `gene_encoder.*`, `transformer_encoder.*` → `transformer.*`, `decoder.fc.*` → `expr_decoder.fc.*`), and the flash-attention `self_attn.Wqkv.{weight,bias}` of every layer equals the torch-layout `self_attn.in_proj_{weight,bias}` exactly.
- The 14 tensors the packager dropped are a 177-class `cls_decoder` (a cell-type head the original checkpoint carried), the `mvc_decoder` (the masked-value-from-cell-embedding pretraining head) and the `flag_encoder`. None of them is needed for embeddings or for the masked-expression decoder; the packaged model is therefore a faithful *subset* of the original, not a re-training.

## The architecture is re-implemented here, and why

`config.json` declares `model_type: "scgpt"` with no `auto_map`; nothing on the Hub can build the model. The packager's loader (`tdc.model_server.models.scgpt.ScGPTModel` in PyTDC) needs `transformers` and `flash_attn` as dependencies and diverges from the original scGPT in ways that matter: its value encoder has **no ReLU** between its two linears (the original's `ContinuousValueEncoder` has one, plus a clamp at 512), its README example feeds **raw expression values** where the checkpoint was trained on 51-level per-cell quantile bins, its tokenizer maps **unknown gene symbols to id 0** (the gene A1BG), and its example passes an attention mask whose polarity is inverted for `src_key_padding_mask`.

`build_model()` therefore follows the original `bowang-lab/scGPT` `model.py` at commit `cebd6fae655b9c585a4807daa3ac31bb764f06b4`: `GeneEncoder` (embedding + LayerNorm), `ContinuousValueEncoder` (clamp 512 → Linear(1,512) → ReLU → Linear → LayerNorm → dropout), a post-norm `nn.TransformerEncoder` with ReLU feed-forward and `batch_first=True` (the same layer the original uses on its non-flash path, and the layout the flash path's weights map onto exactly), `<cls>` pooling, and `ExprDecoder` (Linear → LeakyReLU → Linear → LeakyReLU → Linear(1)). Parameter names match the checkpoint, so `load_state_dict(strict=True)` is the structural proof.

Functional evidence, from the build record (10x Genomics `pbmc3k`, not committed): the frozen `<cls>` embeddings separate marker-defined B, T and monocyte cells at 72/72 leave-one-out nearest centroid after centring (centred cosine 0.427 within group versus −0.224 between), and the ReLU variant predicts masked bins better than the no-ReLU variant (Pearson 0.41 versus 0.34 against the tie-spread expectation). What was **not** done: a numerical-parity comparison with the upstream `scgpt` package, which was not installed (it pins an older torch and flash-attention). A DIMER profile that needs bit-level agreement with the upstream package must add that check.

## Input encoding follows `scgpt/preprocess.py`

Detected genes are normalised to 10,000 total counts and log1p-transformed, and the non-zero values are digitised against 50 per-cell quantile edges into bins 1–50 (zeros are never tokenised). Upstream's `_digitize(side="both")` spreads tied values **uniformly at random** between their left and right bin — sparse count data has many ties (every count-of-1 gene shares one value) and the checkpoint was trained on that spread — so `bin_expression` does the same with a seeded generator (`BINNING_SEED = 42`, digitising in ascending token-id order): identical input and seed give identical bins, and the distribution matches upstream's. Ranking, normalisation and log1p are all monotone within a cell, so raw and normalised counts produce the same bins. Cells with more than 1,535 detected genes keep the most expressed and report the truncation; masked positions carry `-1`, the training-time mask value.

## The vocabulary: a second source, pinned the same way

`vocab.json` (1,317,639 bytes, SHA-256 `ee2b2c90158eedb97c2318e49abaaed0a02c6fdf7e3f7ca6a821906413c4d2a4`) is TDC's `scgpt_vocab`: 60,697 entries mapping HGNC gene symbols to token ids 0..60696 with `<pad>` 60694, `<cls>` 60695 and `<eoc>` 60696 — the size `config.json` declares. It is **not** in the Hub repository; PyTDC downloads it at runtime from Harvard Dataverse. The manifest lists it as a fourth entry with `source: https://dataverse.harvard.edu/api/access/datafile/10809431` (a persistent file id) and its digest; `stage_missing_files` tries that source first and, if unavailable, the immutable mirror at repository revision `677560540f84664db00af8e03a924c65f0ca4e7a`, then `verify_snapshot` checks the same declared byte size and SHA-256 regardless of source. It is also committed in `weights/scgpt/` so a clone works offline. Two facts a reader should know: `config.json` says `pad_token_id: 0` while the vocabulary's `<pad>` is 60694 — this package uses the vocabulary's value, as the original scGPT does (`padding_idx=vocab["<pad>"]`); and id 0 is the gene `A1BG`, which is why unknown symbols are dropped and reported here rather than mapped to 0.

## Files deliberately not staged

The upstream repository at the pinned revision also carries `scgpt_gh_repo_original_model.bin` (207,861,754 bytes; a pickle, compared at build time as described above and never loaded by the pipeline) and `.gitattributes` (1,589 bytes). Neither is listed in the manifest. A DIMER profile upload for this model should use `model.safetensors` + `config.json` + `vocab.json`, and must not upload the `.bin`.

## DIMER hosting

- MIT permits use, modification, redistribution and commercial use subject to preservation of the licence and copyright notice. DIMER may mirror the pinned snapshot in its model store under those terms; the weights would be redistributed unmodified.
- Loader trust boundary: no `trust_remote_code`, no third-party loader, no Hub or Dataverse access on the snapshot path once staged. torch and safetensors are the runtime; there is no transformers version constraint at all.
- Serving shape: embeddings need the encoder and the vocabulary; an adapted profile needs the encoder plus a 6 MB (one unfrozen layer) or 6 KB (head-only) adapter. The masked-expression decoder should not be advertised: on the data tried it predicts a near-constant bin.
- Contract: the fleet inventory's nominated first contract for this row is multi-batch integration, which this repository does not implement (the packaged checkpoint also lacks the original's `dsbn` batch machinery). Embeddings, a zero-training centroid classifier and a bounded classifier are what is exposed; the encoder is small enough (1.8 s to load, ~40 ms per cell) to serve on CPU.
- Line endings: `.gitattributes` carries `weights/** -text`, so a Windows checkout cannot rewrite a snapshot file's newlines and break its recorded digest — this matters doubly here because `vocab.json` is committed.
