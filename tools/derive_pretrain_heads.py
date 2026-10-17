"""Derive ``weights/scgpt/pretrain_heads.safetensors`` from the original scGPT checkpoint (maintainer tool, run once).

The TDC safetensors packaging (``tdc/scGPT`` at ``acf749f``) keeps the encoder and the expression decoder but drops the
two pretraining heads the generative masked-expression call needs: the ``flag_encoder`` (the "this gene is to be
generated" embedding) and the ``mvc_decoder`` (the masked-value-from-cell-embedding head). Both survive only in
``scgpt_gh_repo_original_model.bin``, a pickle in the same Hub repository at the same immutable revision. The pipeline
and the notebook never load a pickle, so this script, run once in a scratch environment, follows the repository's
pickle-handling procedure (``docs/WEIGHTS.md``) and writes the four needed tensors to safetensors:

1. the ``.bin`` must have the pinned byte size and SHA-256 (the Hub LFS ``oid`` at the pinned revision);
2. every pickle inside the zip archive gets a static pre-check with ``pickletools`` (no code is executed): each global it
   references must be in a small allowlist of tensor-rebuild helpers. It tracks preceding string arguments rather than
   emulating the pickle stack, so it is a pre-check, not a security audit; the pinned digest and step 3 are the controls;
3. the archive is then loaded with ``torch.load(weights_only=True)``;
4. only ``flag_encoder.weight``, ``mvc_decoder.W.weight``, ``mvc_decoder.gene2query.weight`` and
   ``mvc_decoder.gene2query.bias`` are written, as little-endian float32, with the source recorded in the
   safetensors metadata. The header is serialised here with sorted keys (the safetensors library orders header keys
   by hash, so its bytes vary between runs), which makes re-derivation byte-identical; the file is then read back
   with ``safetensors`` and compared tensor by tensor. The output size and SHA-256 are printed for the manifest.

Usage (needs torch and safetensors, which CI does not install)::

    python -I tools/derive_pretrain_heads.py <scgpt_gh_repo_original_model.bin> <output_dir>
"""
# ruff: noqa: E501  -- docstring and report lines are kept whole
from __future__ import annotations

import hashlib
import json
import pickletools
import sys
import zipfile
from pathlib import Path

SOURCE_REPO = "tdc/scGPT"
SOURCE_REVISION = "acf749f35bf5c0b00633838f02588272ed0d9911"
SOURCE_FILE = "scgpt_gh_repo_original_model.bin"
SOURCE_BYTES = 207_861_754
SOURCE_SHA256 = "ad0252a1971e0cd619b7116dbab3177432236c4537225d54280a2aa7e5fe402a"
OUTPUT_NAME = "pretrain_heads.safetensors"
TENSORS = (
    "flag_encoder.weight",
    "mvc_decoder.W.weight",
    "mvc_decoder.gene2query.bias",
    "mvc_decoder.gene2query.weight",
)
ALLOWED_GLOBALS = {
    ("collections", "OrderedDict"),
    ("torch._utils", "_rebuild_tensor_v2"),
    ("torch._utils", "_rebuild_parameter"),
    ("torch", "FloatStorage"),
    ("torch", "HalfStorage"),
    ("torch", "LongStorage"),
}


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_pickles(path: Path) -> list[tuple[str, str]]:
    """Return every global the archive's pickles reference; raise if one is outside ALLOWED_GLOBALS."""
    found: set[tuple[str, str]] = set()
    with zipfile.ZipFile(path) as archive:
        pickles = [n for n in archive.namelist() if n.endswith(".pkl")]
        if not pickles:
            raise SystemExit(f"{path.name}: no pickle found in the archive; not a torch zip checkpoint")
        for name in pickles:
            strings: list[str] = []
            for opcode, arg, _ in pickletools.genops(archive.read(name)):
                if isinstance(arg, str):
                    strings.append(arg)
                if opcode.name == "GLOBAL":
                    module, _, attr = str(arg).partition(" ")
                    found.add((module, attr))
                elif opcode.name == "STACK_GLOBAL":
                    if len(strings) < 2:
                        raise SystemExit(f"{name}: STACK_GLOBAL without two preceding string arguments")
                    found.add((strings[-2], strings[-1]))
                elif opcode.name in {"INST", "OBJ", "EXT1", "EXT2", "EXT4"}:
                    raise SystemExit(f"{name}: opcode {opcode.name} is not allowed in a weights pickle")
    unexpected = sorted(found - ALLOWED_GLOBALS)
    if unexpected:
        raise SystemExit(f"{path.name}: pickle references globals outside the allowlist: {unexpected}")
    return sorted(found)


def write_safetensors(path: Path, arrays: dict, metadata: dict[str, str]) -> None:
    """Deterministic safetensors writer: sorted header keys, tensors in name order, 8-byte aligned header."""
    header: dict[str, object] = {"__metadata__": metadata}
    blobs, offset = [], 0
    for name in sorted(arrays):
        data = arrays[name].astype("<f4", copy=False).tobytes(order="C")
        header[name] = {"dtype": "F32", "shape": list(arrays[name].shape), "data_offsets": [offset, offset + len(data)]}
        blobs.append(data)
        offset += len(data)
    encoded = json.dumps(header, sort_keys=True, separators=(",", ":")).encode("utf-8")
    encoded += b" " * (-len(encoded) % 8)
    path.write_bytes(len(encoded).to_bytes(8, "little") + encoded + b"".join(blobs))


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        raise SystemExit(__doc__)
    source, out_dir = Path(argv[0]), Path(argv[1])
    size = source.stat().st_size
    digest = sha256_of(source)
    if (size, digest) != (SOURCE_BYTES, SOURCE_SHA256):
        raise SystemExit(f"{source}: {size} bytes, sha256 {digest}; expected {SOURCE_BYTES}, {SOURCE_SHA256}")
    globals_seen = audit_pickles(source)

    import torch
    from safetensors.torch import load_file

    state = torch.load(str(source), map_location="cpu", weights_only=True)
    missing = [k for k in TENSORS if k not in state]
    if missing:
        raise SystemExit(f"{source.name} lacks {missing}")
    heads = {k: state[k].detach().to(torch.float32).contiguous().clone() for k in TENSORS}
    if any(t.dtype != torch.float32 for t in (state[k] for k in TENSORS)):
        raise SystemExit("a source tensor is not float32; refusing to convert silently")
    metadata = {
        "source_repo": SOURCE_REPO,
        "source_revision": SOURCE_REVISION,
        "source_file": SOURCE_FILE,
        "source_sha256": SOURCE_SHA256,
        "derived_by": "tools/derive_pretrain_heads.py",
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / OUTPUT_NAME
    write_safetensors(out, {k: v.numpy() for k, v in heads.items()}, metadata)
    reread = load_file(str(out))
    if sorted(reread) != sorted(TENSORS) or not all(torch.equal(reread[k], heads[k]) for k in TENSORS):
        raise SystemExit(f"{out}: read-back does not reproduce the source tensors")
    report = {
        "source": {"repo": SOURCE_REPO, "revision": SOURCE_REVISION, "file": SOURCE_FILE, "bytes": size, "sha256": digest},
        "pickle_globals": [f"{m}.{a}" for m, a in globals_seen],
        "tensors": {k: list(v.shape) for k, v in heads.items()},
        "output": {"path": out.name, "bytes": out.stat().st_size, "sha256": sha256_of(out)},
    }
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
