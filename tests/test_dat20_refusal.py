"""DAT19/DAT20: a refused BYOD input shows exactly one `Refused: <file>: <reason>` line and stops Run all.

Runs the committed kernel `run_stage` (from the notebook's install cell) against the carried stage runner in a
subprocess, as the notebook does. Covers the scripted hosted refusal input (`BYOD_PATH='/etc/hostname'`:
a file with no dataset suffix), a directory, a cancelled upload, the later-stage stop and a genuine failure.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "tutorials" / "scgpt_single_cell_colab.ipynb"
REFUSAL_CELL = "scgpt_single_cell-15"


def _code_cells() -> list[dict]:
    return [c for c in json.loads(NB.read_text(encoding="utf-8"))["cells"] if c["cell_type"] == "code"]


@pytest.fixture
def kernel(tmp_path):
    root, weights = tmp_path / "run", tmp_path / "weights"
    shutil.copytree(ROOT / "src", root / "src", ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copy(ROOT / "tools" / "tutorial_stages.py", root / "tutorial_stages.py")
    shutil.copytree(
        ROOT / "weights" / "scgpt",
        weights / "scgpt",
        ignore=shutil.ignore_patterns("*.safetensors", "*.pt", "*.bin"),
    )
    install = next("".join(c["source"]) for c in _code_cells() if "def run_stage(" in "".join(c["source"]))
    block = install[install.index("class StageRefusal(") : install.index("def load_record(")]
    ns = {"ROOT": root, "OUTPUTS": tmp_path / "outputs", "WEIGHTS": weights, "PYTHON": Path(sys.executable)}
    ns.update(ENV=dict(os.environ, CUDA_VISIBLE_DEVICES="-1"), json=json, subprocess=subprocess)
    exec(compile(block, "<install>", "exec"), ns)
    return ns


def _refused(kernel, capsys, **options) -> str:
    with pytest.raises(kernel["StageRefusal"]) as caught:
        kernel["run_stage"]("data", use_byod=True, **options)
    assert caught.value.dimer_refusal is True
    rendered = caught.value._render_traceback_()
    printed = capsys.readouterr().out
    assert "Traceback" not in printed and "refused" not in printed.lower(), (
        printed
    )  # the stage does not echo it
    assert len(rendered) == 1 and "\n" not in rendered[0], rendered
    assert rendered[0].startswith("Refused: ") and "Traceback" not in rendered[0]
    return rendered[0]


def test_scripted_hosted_input_a_file_with_no_dataset_suffix(kernel, capsys, tmp_path):
    hostname = tmp_path / "hostname"
    hostname.write_text("colab-vm\n")
    line = _refused(kernel, capsys, byod_path=str(hostname))
    assert line.startswith("Refused: hostname: BYOD must be a genes-as-columns .csv"), line
    assert "Nothing was run on it. To recover: fix the named file" in line


def test_a_directory_and_a_cancelled_upload(kernel, capsys, tmp_path):
    line = _refused(kernel, capsys, byod_path=str(tmp_path))
    assert line.startswith(f"Refused: {tmp_path}: BYOD_PATH is not a file in this runtime"), line
    cancelled = "(no file): the upload was cancelled or empty: no file was received."
    line = _refused(kernel, capsys, byod_path="", byod_refusal=cancelled)
    assert line.startswith("Refused: (no file): the upload was cancelled or empty"), line


def test_later_stages_stop_cleanly_after_a_refusal(kernel, capsys, tmp_path):
    _refused(kernel, capsys, byod_path=str(tmp_path / "missing.csv"))
    with pytest.raises(kernel["StageNotRun"]) as caught:
        kernel["run_stage"]("tokens")
    rendered = caught.value._render_traceback_()
    assert len(rendered) == 1 and rendered[0].startswith("Skipped: 'tokens' did not run"), rendered


def test_genuine_failure_keeps_runtime_error(kernel):
    with pytest.raises(RuntimeError, match="failed"):
        kernel["run_stage"]("no-such-stage")


def test_refusal_cell_id():
    ids = [c["id"] for c in _code_cells() if "run_stage('data'" in "".join(c["source"])]
    assert ids == [REFUSAL_CELL]
