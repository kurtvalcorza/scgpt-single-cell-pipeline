"""Regression tests for the 2026-10-05 notebook review (SCG-M1, SCG-M2, SCG-m1, SCG-m2, SCG-S1, SCG-S2).

They need only CI's dependencies (no torch, no numpy, no checkpoint): they exec the notebook's own kernel cells with
stand-ins for `run_stage` and `google.colab`, run the stage runner's model-free stages (`data`, `tokens`) in-process with
the committed vocabulary, and check the generated notebook statically. Each test names its finding.
"""
# ruff: noqa: E501

from __future__ import annotations

import ast
import contextlib
import csv
import hashlib
import importlib.util
import io
import json
import re
import shutil
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
NB = ROOT / "tutorials" / "scgpt_single_cell_colab.ipynb"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


STAGES = _load("scg_tutorial_stages", TOOLS / "tutorial_stages.py")


def _nb() -> dict:
    return json.loads(NB.read_text(encoding="utf-8"))


def _src(cell: dict) -> str:
    return "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]


def _cell_with(needle: str) -> str:
    found = [_src(c) for c in _nb()["cells"] if c["cell_type"] == "code" and needle in _src(c)]
    assert len(found) == 1, needle
    return found[0]


def _set(src: str, name: str, value) -> str:
    out = []
    for line in src.split("\n"):
        if line.startswith(f"{name} = "):
            line = f"{name} = {value!r}" + (line[line.index("  #"):] if "  #" in line else "")
        out.append(line)
    return "\n".join(out)


@contextlib.contextmanager
def _colab(queue):
    files = types.SimpleNamespace(calls=0)

    def upload():
        files.calls += 1
        return queue.pop(0)

    files.upload = upload
    colab = types.ModuleType("google.colab")
    colab.files = files
    google = types.ModuleType("google")
    google.colab = colab
    saved = {k: sys.modules.get(k) for k in ("google", "google.colab")}
    sys.modules.update({"google": google, "google.colab": colab})
    try:
        yield files
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v


def _no_colab(monkeypatch):
    monkeypatch.setitem(sys.modules, "google.colab", None)


def _kernel(tmp_path: Path) -> tuple[dict, list]:
    calls: list = []
    ns = {"ROOT": tmp_path / "run", "Path": Path, "shutil": shutil, "run_stage": lambda stage, **o: calls.append((stage, o))}
    return ns, calls


def _run(tmp_path: Path, options: dict | None = None):
    root = tmp_path / "run"
    if not (root / "src").exists():
        root.mkdir(parents=True, exist_ok=True)
        (root / "src").symlink_to(ROOT / "src", target_is_directory=True)
    weights = tmp_path / "weights"
    if not (weights / "scgpt").exists():
        weights.mkdir(parents=True, exist_ok=True)
        (weights / "scgpt").symlink_to(ROOT / "weights" / "scgpt", target_is_directory=True)
    return STAGES.Run(root, tmp_path / "outputs", weights, options or {})


def _quiet(fn, *args):
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        fn(*args)
    return out.getvalue()


def _sample_records() -> list[dict]:
    sys.path.insert(0, str(ROOT / "src"))
    import scgpt_single_cell_pipeline as P

    return P.generate_sample_dataset(P.load_gene_vocabulary(ROOT / "weights" / "scgpt"))


def _write_csv(path: Path, records: list[dict]) -> Path:
    genes = sorted({g for r in records for g in r["counts"]})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", *genes, "label"])
        for r in records:
            writer.writerow([r["id"], *[r["counts"].get(g, "") for g in genes], r["label"]])
    return path


# ---------------------------------------------------------------- SCG-M1: isolated runtime


def test_M1_no_kernel_install_and_no_restart_instruction() -> None:
    text = NB.read_text(encoding="utf-8")
    assert "Restart the runtime" not in text and "restart the runtime" not in text
    own = [_src(c) for c in _nb()["cells"] if c["cell_type"] == "code" and not c["metadata"].get("dimer", {}).get("embedded_sources")]
    for src in own:
        assert not re.search(r"['\"]-m['\"]\s*,\s*['\"]pip['\"]|^\s*[%!]\s*pip\b|['\"]pip install", src, re.M), src[:80]
        assert "import torch" not in src and "from scgpt_single_cell_pipeline" not in src
    install = _cell_with("# @title Infrastructure: build (or reuse) the isolated")
    assert "'--require-hashes'" in install and "--managed-python" in install
    assert "MPLBACKEND='Agg'" in install and "'PYTHONPATH', 'PYTHONHOME', 'PYTHONSTARTUP'" in install


def _exec_check_cell(ns: dict, **overrides) -> None:
    src = _cell_with("# @title Infrastructure: check the runtime")
    src = re.sub(r"'environment': [0-9.]+}", "'environment': 0.0}", src, count=1)
    src = re.sub(r"\{'weights': max\(0\.0, [0-9.]+", "{'weights': max(0.0, 0.0", src, count=1)
    for name, value in overrides.items():
        src = _set(src, name, value)
    with contextlib.redirect_stdout(io.StringIO()):
        exec(compile(src, "<check>", "exec"), ns)


def test_M1_section1_is_idempotent(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    ns: dict = {}
    _exec_check_cell(ns)
    first = ns["ROOT"]
    (first / "tutorial_stages.py").write_text("# carried")
    _exec_check_cell(ns)
    assert ns["ROOT"] == first and (first / "tutorial_stages.py").is_file()
    _exec_check_cell(ns, NEW_RUN_DIRECTORY=True)
    assert ns["ROOT"] != first


def test_M1_second_run_all_reuses_the_matching_environment(tmp_path, monkeypatch) -> None:
    """SCG-M1: the venv is keyed on the lock digest; a second exec of the install cell builds nothing."""
    import subprocess as real_subprocess

    monkeypatch.chdir(tmp_path)
    ns: dict = {}
    _exec_check_cell(ns)
    ns["ENV_ROOT"] = tmp_path / "uvroot"
    ns["NOTEBOOK_SOURCE"] = {"revision": "test"}
    src = _cell_with("# @title Infrastructure: build (or reuse) the isolated")
    version = re.search(r"UV = ENV_ROOT / 'uv-([0-9.]+)'", src).group(1)
    ns["ENV_ROOT"].mkdir()
    (ns["ENV_ROOT"] / f"uv-{version}").write_bytes(b"uv stand-in")
    (ns["ENV_ROOT"] / f"uv-{version}.sha256").write_text(hashlib.sha256(b"uv stand-in").hexdigest())
    commands: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        commands.append([str(c) for c in cmd])
        if cmd[1] == "venv":
            python = Path(cmd[-1]) / "bin" / "python"
            python.parent.mkdir(parents=True)
            python.write_text("")
        out = json.dumps({"python": "3.12.12", "torch": "x", "safetensors": "x", "numpy": "x", "cuda": False})
        return types.SimpleNamespace(stdout=out + "\n", returncode=0)

    fake = types.SimpleNamespace(run=fake_run, Popen=real_subprocess.Popen, PIPE=real_subprocess.PIPE, STDOUT=real_subprocess.STDOUT)
    for attempt in range(2):
        ns["subprocess"] = fake
        with contextlib.redirect_stdout(io.StringIO()):
            exec(compile(src.replace("import zipfile\n", "import zipfile\nsubprocess = globals()['subprocess']\n", 1), "<install>", "exec"), ns)
        ns["subprocess"] = fake
        assert ns["environment_reused"] is (attempt == 1)
    assert len([c for c in commands if c[1] in ("venv", "pip")]) == 2


# ---------------------------------------------------------------- SCG-M2: guided layer

GUIDED = ("## How to use this notebook", "**Who this notebook is for.**", "## The task: Input → Model → Output", "## Roadmap", "<summary><strong>Glossary</strong>", "Predict before running", "**What to notice:**", "Check your reasoning", "## Troubleshooting", "## Conclusion", "Sample conclusion")


def test_M2_guided_layer_and_collapsed_infrastructure() -> None:
    nb = _nb()
    md = "\n".join(_src(c) for c in nb["cells"] if c["cell_type"] == "markdown")
    for heading in GUIDED:
        assert heading in md, heading
    assert md.count("Predict before running") >= 3  # before Sections 7, 8 and 10
    infra = [c for c in nb["cells"] if c["cell_type"] == "code" and _src(c).startswith("# @title Infrastructure:")]
    assert len(infra) == 4 and all(c["metadata"].get("cellView") == "form" for c in infra)
    assert "Dataverse" in md and "HTTP 504" in md  # troubleshooting covers the vocabulary fallback


def test_M2_trainable_layers_activity_prints_side_by_side() -> None:
    """SCG-M2: 'Optional experiments' became a Predict → Change → Run → Observe → Explain activity with a three-row comparison."""
    cell = _cell_with("RUN_ACTIVITY = False  # @param")
    assert "ACTIVITY_TRAINABLE_LAYERS = 0  # @param" in cell and "run_stage('activity', trainable_layers=ACTIVITY_TRAINABLE_LAYERS)" in cell
    source = (TOOLS / "tutorial_stages.py").read_text(encoding="utf-8")
    body = source[source.index("def stage_activity"):source.index("STAGES = {")]
    assert '"zero-training nearest centroid"' in body and '"default (Section 9)"' in body and '"activity"' in body
    assert 'write_output(f"activity/' in body and "!= before" in body


def test_no_quality_assert_in_the_stage_runner() -> None:
    """SCG-M2 (brief): invariance and parity checks print verdicts; nothing asserts a quality level."""
    tree = ast.parse((TOOLS / "tutorial_stages.py").read_text(encoding="utf-8"))
    assert not [n for n in ast.walk(tree) if isinstance(n, ast.Assert)]
    learner = [_src(c) for c in _nb()["cells"] if c["cell_type"] == "code" and not _src(c).startswith("# @title Infrastructure:")]
    assert not any(re.search(r"^\s*assert\b", s, re.M) for s in learner)


# ---------------------------------------------------------------- SCG-m1: template braces


def test_m1_no_doubled_braces_in_markdown() -> None:
    md = [_src(c) for c in _nb()["cells"] if c["cell_type"] == "markdown"]
    assert not [m[:80] for m in md if "{{" in m or "}}" in m]
    assert any("{id, counts, label}" in m for m in md)


# ---------------------------------------------------------------- SCG-m2: BYOD path, guarded upload, held-back cells


def test_m2_byod_path_works_outside_colab_and_empty_path_is_named(tmp_path, monkeypatch) -> None:
    cell = _cell_with("USE_BYOD = False  # @param")
    _no_colab(monkeypatch)
    ns, calls = _kernel(tmp_path)
    exec(compile(_set(_set(cell, "USE_BYOD", True), "BYOD_PATH", "/d/cells.csv"), "<s4>", "exec"), ns)
    assert calls[0] == ("data", {"use_byod": True, "byod_path": "/d/cells.csv", "val_fraction": 0.2, "test_fraction": 0.25, "seed": 42, "inference_cells": 6})
    ns, calls = _kernel(tmp_path)
    with pytest.raises(RuntimeError, match="BYOD_PATH is empty, and the upload dialog exists only in Google Colab"):
        exec(compile(_set(cell, "USE_BYOD", True), "<s4>", "exec"), ns)
    ns, calls = _kernel(tmp_path)
    exec(compile(cell, "<s4>", "exec"), ns)
    assert calls[0][1]["use_byod"] is False and calls[0][1]["byod_path"] == ""


@pytest.mark.parametrize(("queue", "message"), [([{}], "cancelled or empty"), ([{"a.csv": b"x", "b.csv": b"y"}], "Upload exactly one file")], ids=["cancelled", "two-files"])
def test_m2_upload_is_guarded(tmp_path, queue, message) -> None:
    cell = _set(_cell_with("USE_BYOD = False  # @param"), "USE_BYOD", True)
    with _colab(queue) as files:
        ns, calls = _kernel(tmp_path)
        with pytest.raises((RuntimeError, ValueError), match=message):
            exec(compile(cell, "<s4>", "exec"), ns)
        assert files.calls == 1 and calls == []


def test_m2_byod_inference_cells_are_held_back_before_the_split(tmp_path) -> None:
    """SCG-m2: BYOD 'new' cells are held back before the split, so they are outside train, validation and test."""
    path = _write_csv(tmp_path / "mine.csv", _sample_records()[:40])
    run = _run(tmp_path, {"use_byod": True, "byod_path": str(path), "inference_cells": 6})
    printed = _quiet(STAGES.stage_data, run)
    data = json.loads((run.state / "data.json").read_text())
    split_ids = {r["id"] for part in ("train", "validation", "test") for r in data[part]}
    new_ids = {r["id"] for r in data["new"]}
    assert len(new_ids) == 6 and not (new_ids & split_ids)
    assert len(split_ids) + len(new_ids) == 40
    assert data["source"] == "BYOD (mine.csv)" and "held back before the split" in printed


def test_m2_small_byod_dataset_holds_back_fewer_and_says_so(tmp_path) -> None:
    """SCG-m2: with 8 cells per class only one per class can be held back while the training split stays valid."""
    records = _sample_records()
    small = [r for r in records if r["label"] == "t-like"][:8] + [r for r in records if r["label"] == "b-like"][:8]
    path = _write_csv(tmp_path / "small.csv", small)
    run = _run(tmp_path, {"use_byod": True, "byod_path": str(path), "inference_cells": 6})
    printed = _quiet(STAGES.stage_data, run)
    data = json.loads((run.state / "data.json").read_text())
    assert len(data["new"]) == 2 and "only 2 of 6 cells could be held back while keeping a valid training split" in printed
    assert len(data["train"]) >= 8


def test_m2_too_small_byod_dataset_is_refused_with_the_true_minimum(tmp_path) -> None:
    """SCG-m2 / brief (true minimum): 5 cells per class pass the record-level contract (>= 8 records, >= 3 per class) but
    leave a 6-cell training split that the baselines and adapt would refuse in Section 8 or 9; Section 4 refuses it now,
    naming the minimum."""
    records = _sample_records()
    tiny = [r for r in records if r["label"] == "t-like"][:5] + [r for r in records if r["label"] == "b-like"][:5]
    path = _write_csv(tmp_path / "tiny.csv", tiny)
    run = _run(tmp_path, {"use_byod": True, "byod_path": str(path)})
    with pytest.raises(ValueError, match=r"tiny\.csv: the training split would hold \d+ cells .*supply at least 7 cells per class"):
        _quiet(STAGES.stage_data, run)
    sys.path.insert(0, str(ROOT / "src"))
    import scgpt_single_cell_pipeline as P

    assert [STAGES.min_cells_per_class(P, c, 0.2, 0.25) for c in (2, 3)] == [7, 5]


@pytest.mark.parametrize(
    ("name", "content", "message"),
    [
        ("cells.txt", "id,label\n", r"cells\.txt: BYOD must be a genes-as-columns \.csv"),
        ("ragged.csv", "id,CD3D,CD3E,label\nc1,1,2,t\nc2,1,t\n", r"ragged\.csv: line 3 has 3 fields, header has 4"),
        ("ensembl.csv", "id," + ",".join(f"ENSG{i:011d}" for i in range(12)) + ",label\n" + "".join(f"c{j}," + ",".join("5" for _ in range(12)) + f",{'ab'[j % 2]}\n" for j in range(10)), r"ensembl\.csv: .*are these human gene symbols\?"),
    ],
    ids=["txt", "ragged-row", "ensembl-ids"],
)
def test_m2_refusals_name_the_file_and_the_rule(tmp_path, name, content, message) -> None:
    path = tmp_path / name
    path.write_text(content, encoding="utf-8")
    run = _run(tmp_path, {"use_byod": True, "byod_path": str(path)})
    with pytest.raises(ValueError, match=message):
        _quiet(STAGES.stage_data, run)


def test_m2_missing_byod_file_names_the_field(tmp_path) -> None:
    run = _run(tmp_path, {"use_byod": True, "byod_path": str(tmp_path / "absent.csv")})
    with pytest.raises(FileNotFoundError, match="BYOD_PATH .*absent.csv.* is not a file"):
        _quiet(STAGES.stage_data, run)


# ---------------------------------------------------------------- default path, model-free stages


def test_default_data_and_tokens_stages_match_the_recorded_split(tmp_path) -> None:
    """The default sample: 64 cells, 36 / 12 / 16, six freshly generated new cells outside every split; tokens report a
    scale-invariance verdict and write the input manifest with the four refusal probes."""
    run = _run(tmp_path, {"use_byod": False, "seed": 42})
    _quiet(STAGES.stage_data, run)
    data = json.loads((run.state / "data.json").read_text())
    assert (len(data["train"]), len(data["validation"]), len(data["test"]), len(data["new"])) == (36, 12, 16, 6)
    assert data["classes"] == ["b-like", "t-like"]
    assert not ({r["id"] for r in data["new"]} & {r["id"] for p in ("train", "validation", "test") for r in data[p]})
    printed = _quiet(STAGES.stage_tokens, run)
    assert "'scale_invariance': 'PASS'" in printed
    manifest = json.loads((run.out / "scgpt_single_cell_input_manifest.json").read_text())
    assert manifest["verdict"] == "accepted" and set(manifest["probes"]) == {"empty cell", "two genes", "negative count", "no known symbols"}
    assert all(v != "accepted" for v in manifest["probes"].values())


def test_reload_parity_is_a_contract_check_against_the_adapting_process() -> None:
    """The adapter is rebuilt in fresh processes for Sections 10 and 11; a reload that does not reproduce the adapting
    process's predictions stops the notebook (contract integrity, not a quality threshold)."""
    source = (TOOLS / "tutorial_stages.py").read_text(encoding="utf-8")
    assert "reference_predictions = pipe.classify(" in source[source.index("def stage_adapt"):source.index("def reloaded")]
    body = source[source.index("def stage_export"):source.index("def stage_result")]
    assert "pipe, adapt = reloaded(run, P)" in body and "if not labels_equal or max_diff > RELOAD_TOLERANCE:" in body
    assert "Do not ship this artifact" in body and STAGES.RELOAD_TOLERANCE == 1e-5


# ---------------------------------------------------------------- SCG-S1 / SCG-S2: saturation and the library-size reading


def test_S1_S2_saturation_and_library_size_chance_are_disclosed() -> None:
    md = "\n".join(_src(c) for c in _nb()["cells"] if c["cell_type"] == "markdown")
    assert "the default task is easy on purpose" in md and "cannot show a gain" in md
    assert "p ≈ 0.16" in md and "±12 counts" in md
    source = (TOOLS / "tutorial_stages.py").read_text(encoding="utf-8")
    assert "saturated: the zero-training nearest-centroid rule is already perfect" in source
