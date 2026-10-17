"""Regression tests for the 2026-10-17 tutorial review (row 80, findings T80-1..T80-6).

CI installs no torch, so these tests check the derived pretraining-heads file without torch (the safetensors header is
plain JSON), run the model-free stages for the BYOD refusal path, and check the gating of verification checks in source.
"""
# ruff: noqa: E501  -- long assertion lines keep each check on one line
from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import io
import json
import re
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = ROOT / "weights" / "scgpt"
NB = ROOT / "tutorials" / "scgpt_single_cell_colab.ipynb"
sys.path.insert(0, str(ROOT / "src"))

import scgpt_single_cell_pipeline.pipeline as P  # noqa: E402


def _load_stages():
    spec = importlib.util.spec_from_file_location("scg_stages_t80", ROOT / "tools" / "tutorial_stages.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


STAGES = _load_stages()


def _cells() -> list[str]:
    return ["".join(c["source"]) for c in json.loads(NB.read_text(encoding="utf-8"))["cells"]]


# ---------------------------------------------------------------- T80-1: pinned pretraining heads, generative call


def test_t80_1_pretrain_heads_are_a_pinned_manifest_entry() -> None:
    manifest = json.loads((WEIGHTS / P.MANIFEST_NAME).read_text(encoding="utf-8"))
    entry = next(e for e in manifest["files"] if e["path"] == P.PRETRAIN_HEADS_NAME)
    assert len(manifest["files"]) == 5 and manifest["totalBytes"] == sum(e["bytes"] for e in manifest["files"])
    assert entry["derived_from"] == {"modelId": P.MODEL_ID, "revision": P.MODEL_REVISION, **P.PRETRAIN_HEADS_SOURCE}
    assert entry["source"] == P.PRETRAIN_HEADS_MIRROR_URL
    assert re.search(r"/[0-9a-f]{40}/weights/scgpt/pretrain_heads\.safetensors$", P.PRETRAIN_HEADS_MIRROR_URL)
    data = (WEIGHTS / P.PRETRAIN_HEADS_NAME).read_bytes()
    assert len(data) == entry["bytes"] and hashlib.sha256(data).hexdigest() == entry["sha256"]
    n = int.from_bytes(data[:8], "little")
    header = json.loads(data[8 : 8 + n])
    meta = header.pop("__metadata__")
    assert meta["source_sha256"] == P.PRETRAIN_HEADS_SOURCE["sha256"] and meta["source_revision"] == P.MODEL_REVISION
    assert {k: tuple(v["shape"]) for k, v in header.items()} == P.PRETRAIN_HEADS_TENSORS
    assert {v["dtype"] for v in header.values()} == {"F32"}


def test_t80_1_generative_call_needs_a_loaded_model() -> None:
    pipe = P.ScGPTPipeline(lambda rows: [], P.load_gene_vocabulary(WEIGHTS), "cpu")
    with pytest.raises(RuntimeError, match="requires a pipeline built by from_pretrained"):
        pipe.predict_masked_generative([{"CD3D": 1.0}])
    pipe.model = object()
    with pytest.raises(RuntimeError, match="pretrain_heads.safetensors"):
        pipe.predict_masked_genes_generative([{"CD3D": 1.0}], ["CD3D"])


def test_t80_1_probe_text_names_the_data_cause_and_the_conventions() -> None:
    text = "\n".join(_cells())
    assert "training-time mask value" not in text and "property of the shipped decoder" not in text
    assert "chance-level" not in text and "does not use lineage context" not in text
    assert "code-generated" in _cells()[25] and "not an imputer" in _cells()[25] and "26–33" in _cells()[25]
    assert "predict_masked_generative" in _cells()[23] and "no command line" in _cells()[23]
    src = (ROOT / "tools" / "tutorial_stages.py").read_text(encoding="utf-8")
    assert "pipe.predict_masked_generative(" in src and "pipe.predict_masked_genes_generative(" in src


# ---------------------------------------------------------------- T80-2..T80-4: text and metadata


def test_t80_2_pretraining_overlap_is_stated() -> None:
    cells = _cells()
    assert re.search(r"CELLxGENE Census release 2023-05-08, which includes blood", cells[0])
    assert any("overlap" in c and "pretrain" in c for c in cells[30:])  # Interpretation and limits


def test_t80_3_quotes_match_the_recorded_run() -> None:
    cells = _cells()
    assert "recorded Colab CLI T4 run" in cells[1] and "Kaggle T4 run" not in cells[1]
    assert "504 during the recorded run" not in cells[13]
    assert "9.12 s" in cells[31] and "about 7 s" not in cells[31]
    assert "27.04" in cells[25] and "32.02" in cells[25]
    assert "not re-run here" in cells[28] and "docs/reviews/2026-10-05-notebook-review" in cells[28]


def test_t80_4_spec_kind_notice_and_ieee_references() -> None:
    nb = json.loads(NB.read_text(encoding="utf-8"))
    meta = nb["metadata"]["dimer"]
    assert meta["notebook_spec"] == "2.5" and meta["notebook_kind"] == "TUTORIAL"
    first_code = next(i for i, c in enumerate(nb["cells"]) if c["cell_type"] == "code")
    notices = [c for c in _cells()[1:first_code] if c.startswith("> **⚠️ Notice.**")]
    assert len(notices) == 1 and "for clinical or diagnostic use" in notices[0]
    refs = _cells()[-1].split("## References", 1)[1]
    assert re.search(r"^\[1\] H\. Cui, .*doi: 10\.1038/s41592-024-02201-0", refs, re.M)
    assert all(f"[{k}]" in "\n".join(_cells()) for k in range(1, 5))


# ---------------------------------------------------------------- T80-5: verification checks gate success


def test_t80_5_embedding_and_parity_checks_raise() -> None:
    src = (ROOT / "tools" / "tutorial_stages.py").read_text(encoding="utf-8")
    embed = src[src.index("def stage_embed(") : src.index("def stage_probe(")]
    assert re.search(r'if "FAIL" in checks\.values\(\):\s*\n\s*raise RuntimeError\(f"Embedding verification failed', embed)
    assert embed.index("raise RuntimeError") < embed.index('run.write_state("embed.json"')
    evaluate = src[src.index("def stage_evaluate(") : src.index("def stage_export(")]
    gate = evaluate.index('if report["max_abs_metric_difference_vs_adapting_process"] != 0.0:')
    assert gate < evaluate.index("run.write_output(") and "Evaluation parity failed" in evaluate


# ---------------------------------------------------------------- T80-6: BYOD refusal (DAT20)


def _main(tmp_path: Path, stage: str, options: dict) -> tuple[int, str, str]:
    root = tmp_path / "run"
    if not (root / "src").exists():
        shutil.copytree(ROOT / "src", root / "src", ignore=shutil.ignore_patterns("__pycache__"))
        (tmp_path / "weights" / "scgpt").mkdir(parents=True)
        shutil.copy(WEIGHTS / P.VOCAB_NAME, tmp_path / "weights" / "scgpt")
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = STAGES.main(["--root", str(root), "--outputs", str(tmp_path / "outputs"), "--weights", str(tmp_path / "weights"), "--stage", stage, "--options", json.dumps(options)])
    return code, out.getvalue(), err.getvalue()


def test_t80_6_refusal_keeps_outputs_invalidates_the_run_and_shows_no_traceback(tmp_path) -> None:
    code, _, _ = _main(tmp_path, "data", {})
    dataset = tmp_path / "outputs" / f"{STAGES.STEM}_dataset.csv"
    assert code == 0 and dataset.is_file()
    bad = tmp_path / "cells.txt"
    bad.write_text("id,label\n", encoding="utf-8")
    code, out, err = _main(tmp_path, "data", {"use_byod": True, "byod_path": str(bad)})
    assert code == STAGES.REFUSED_EXIT and "Traceback" not in out + err
    assert out.count("BYOD input refused: cells.txt: BYOD must be a genes-as-columns .csv") == 1
    assert "run this cell again" in out and "marked invalid" in out
    assert dataset.is_file() and (tmp_path / "outputs" / f"{STAGES.STEM}_RUN_INVALIDATED.txt").is_file()
    error = json.loads((tmp_path / "run" / "state" / "data.error.json").read_text(encoding="utf-8"))
    assert error["refusal"] is True
    code, out, err = _main(tmp_path, "tokens", {})
    assert code == STAGES.REFUSED_EXIT and out.startswith("Skipped: 'tokens' did not run") and "Traceback" not in out + err
    code, _, _ = _main(tmp_path, "data", {"use_byod": True, "byod_path": str(tmp_path / "absent.csv")})
    assert code == STAGES.REFUSED_EXIT
    code, _, _ = _main(tmp_path, "data", {})
    assert code == 0 and not (tmp_path / "outputs" / f"{STAGES.STEM}_RUN_INVALIDATED.txt").exists()
    assert not (tmp_path / "run" / "state" / STAGES.INVALIDATED).exists()


def test_t80_6_unexpected_errors_keep_their_traceback(tmp_path, monkeypatch) -> None:
    def broken(run):
        raise KeyError("boom")

    monkeypatch.setitem(STAGES.STAGES, "tokens", broken)
    code, out, err = _main(tmp_path, "tokens", {})
    assert code == 1 and "Traceback" in err and "KeyError: 'boom'" in err


def test_t80_6_kernel_does_not_raise_on_a_refusal() -> None:
    carrier = next(c for c in _cells() if "def run_stage(stage, **options):" in c)
    assert "if error.get('refusal') and process.returncode == 3:" in carrier
    assert STAGES.REFUSED_EXIT == 3
