"""Import-boundary contract (fleet RTM-001).

Rejected requests never import model libraries; the snapshot — including the Dataverse-sourced
vocabulary — is verified before anything is imported.
"""

import hashlib
import json

import pytest

from scgpt_single_cell_pipeline.pipeline import (
    CONFIG_NAME,
    MANIFEST_NAME,
    MODEL_ID,
    MODEL_REVISION,
    VOCAB_NAME,
    WEIGHTS_NAME,
    GeneVocabulary,
    ScGPTPipeline,
    validate_inputs,
)

_CONFIG = json.dumps({"model_type": "scgpt", "embsize": 512}).encode()
_VOCAB = GeneVocabulary(tokens={"GENE1": 0, "<pad>": 1, "<cls>": 2, "<eoc>": 3}, pad_id=1, cls_id=2, eoc_id=3)


def _snapshot(root, tamper=False):
    files = []
    for name in (CONFIG_NAME, WEIGHTS_NAME, VOCAB_NAME):
        (root / name).write_bytes(_CONFIG)
        digest = "0" * 64 if (tamper and name == CONFIG_NAME) else hashlib.sha256(_CONFIG).hexdigest()
        files.append({"path": name, "bytes": len(_CONFIG), "sha256": digest})
    manifest = {"modelId": MODEL_ID, "revision": MODEL_REVISION, "files": files}
    (root / MANIFEST_NAME).write_text(json.dumps(manifest), encoding="utf-8")


def test_from_pretrained_refuses_without_manifest_before_model_imports(tmp_path, forbid_model_imports):
    with pytest.raises(FileNotFoundError, match="no snapshot manifest"):
        ScGPTPipeline.from_pretrained(device="cpu", weights_dir=tmp_path, allow_download=False)


def test_from_pretrained_refuses_tampered_snapshot_before_model_imports(tmp_path, forbid_model_imports):
    _snapshot(tmp_path, tamper=True)
    with pytest.raises(ValueError, match="sha256"):
        ScGPTPipeline.from_pretrained(device="cpu", weights_dir=tmp_path, allow_download=False)


def test_config_drift_is_refused_before_model_imports(tmp_path, forbid_model_imports):
    _snapshot(tmp_path)
    with pytest.raises(ValueError, match="disagrees with the package constants"):
        ScGPTPipeline.from_pretrained(device="cpu", weights_dir=tmp_path, allow_download=False)


def test_invalid_cells_are_rejected_before_model_imports(forbid_model_imports):
    with pytest.raises(ValueError, match="at least 10 are required"):
        validate_inputs([{"GENE1": 5}], _VOCAB)
