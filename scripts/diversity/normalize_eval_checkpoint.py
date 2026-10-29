#!/usr/bin/env python3
"""Rename FSDP checkpoint keys for strict loading by the unchanged eval sampler.

This is a CPU-only conversion. It preserves the selected root weights key and
every tensor's values, dtype, shape, and object during serialization.
"""
from __future__ import annotations

import argparse
from collections.abc import Mapping
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile


SCHEMA = "eval_checkpoint_key_normalization_v1"
SOURCE_PREFIX = "model._fsdp_wrapped_module."
DESTINATION_PREFIX = "model."
EXPECTED_COUNT = 825


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def official_sha256(source, supplied):
    if supplied is None:
        sidecar = source.parent / "source.json"
        if sidecar.is_file():
            data = json.loads(sidecar.read_text())
            candidates = {data[key] for key in ("sha256", "expected_sha256", "lfs_sha256", "official_sha256")
                          if isinstance(data.get(key), str)}
            if len(candidates) == 1:
                supplied = candidates.pop()
    digest = str(supplied or "").removeprefix("sha256:").lower()
    if not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ValueError("Provide --source-sha256 with the official SHA-256, or a matching source.json field")
    return digest


def renamed_weights(state, reference, torch):
    if not isinstance(state, Mapping) or not isinstance(reference, Mapping):
        raise ValueError("Selected source and reference weights must be mappings")
    if len(state) != EXPECTED_COUNT or len(reference) != EXPECTED_COUNT:
        raise ValueError(f"Source and reference must each have exactly {EXPECTED_COUNT} tensor keys")
    renamed, mapping = {}, []
    for key, value in state.items():
        if not isinstance(key, str) or not key.startswith(SOURCE_PREFIX) or key == SOURCE_PREFIX:
            raise ValueError(f"Unexpected source key: {key!r}")
        target = DESTINATION_PREFIX + key[len(SOURCE_PREFIX):]
        if target in renamed:
            raise ValueError(f"Normalization key collision: {target}")
        if not isinstance(value, torch.Tensor):
            raise ValueError(f"Source value is not a tensor: {key}")
        if target not in reference or not isinstance(reference[target], torch.Tensor):
            raise ValueError(f"Unexpected normalized key or invalid reference tensor: {target}")
        if tuple(value.shape) != tuple(reference[target].shape):
            raise ValueError(f"Shape mismatch for {target}: {tuple(value.shape)} vs {tuple(reference[target].shape)}")
        if value.device.type != "cpu":
            raise ValueError(f"Expected CPU mmap tensor: {key}")
        renamed[target] = value
        mapping.append(dict(source_key=key, destination_key=target, shape=list(value.shape),
                            dtype=str(value.dtype), reference_dtype=str(reference[target].dtype)))
    if set(renamed) != set(reference):
        raise ValueError("Normalized keys differ from the complete reference generator key set")
    if any(renamed[item["destination_key"]] is not state[item["source_key"]] for item in mapping):
        raise AssertionError("Normalization must retain the original tensor objects")
    return renamed, mapping


def verify_tensors(original, reloaded, torch):
    if set(original) != set(reloaded):
        raise ValueError("Saved checkpoint keys changed")
    for key, before in original.items():
        after = reloaded[key]
        if not isinstance(after, torch.Tensor) or before.dtype != after.dtype or before.shape != after.shape:
            raise ValueError(f"Saved tensor shape/dtype changed: {key}")
        # Compare bits, including signed zeros or any NaN payloads. Contiguous
        # views are used only for verification; the serialized tensors are unchanged.
        a = before.detach().contiguous().reshape(-1).view(torch.uint8)
        b = after.detach().contiguous().reshape(-1).view(torch.uint8)
        if not torch.equal(a, b):
            raise ValueError(f"Saved tensor contents changed: {key}")


def normalize(source, destination, weights, reference_checkpoint, metadata, source_sha256):
    source, destination, reference_checkpoint, metadata = (
        Path(path).resolve() for path in (source, destination, reference_checkpoint, metadata))
    if len({source, destination, reference_checkpoint, metadata}) != 4:
        raise ValueError("Source, destination, reference, and metadata paths must be distinct")
    if not source.is_file() or not reference_checkpoint.is_file():
        raise ValueError("Source and reference checkpoint files must exist")
    expected_source_hash = official_sha256(source, source_sha256)
    print("Verifying official source SHA-256", flush=True)
    source_hash = file_hash(source)
    if source_hash != expected_source_hash:
        raise ValueError(f"Official source SHA-256 mismatch: {source}")
    reference_hash = file_hash(reference_checkpoint)
    if destination.exists() or metadata.exists():
        if not destination.is_file() or not metadata.is_file():
            raise ValueError("Both derived checkpoint and normalization metadata must exist to resume; refusing overwrite")
        previous = json.loads(metadata.read_text())
        expected_identity = dict(schema_version=SCHEMA, weights=weights, source_prefix=SOURCE_PREFIX,
                                 destination_prefix=DESTINATION_PREFIX, tensor_count=EXPECTED_COUNT)
        if any(previous.get(key) != value for key, value in expected_identity.items()):
            raise ValueError("Existing normalization metadata has a different conversion identity")
        for name, path, digest in (("source", source, source_hash),
                                   ("reference", reference_checkpoint, reference_hash),
                                   ("destination", destination, file_hash(destination))):
            recorded = previous.get(name, {})
            if recorded.get("path") != str(path) or recorded.get("sha256") != digest:
                raise ValueError(f"Existing normalization {name} path/hash differs")
        checks = previous.get("validation", {})
        if (len(previous.get("key_mapping", [])) != EXPECTED_COUNT
                or any(checks.get(key) is not True for key in
                       ("strict_reference_keys", "strict_reference_shapes", "tensor_objects_preserved",
                        "reload_bitwise_equal", "dtype_preserved", "rename_only"))):
            raise ValueError("Existing metadata does not establish complete rename-only validation")
        print(json.dumps(dict(status="verified_existing", destination=str(destination), metadata=str(metadata))), flush=True)
        return previous
    import torch

    print("Loading source and reference on CPU with mmap and weights_only=True", flush=True)
    checkpoint = torch.load(source, map_location="cpu", mmap=True, weights_only=True)
    reference = torch.load(reference_checkpoint, map_location="cpu", mmap=True, weights_only=True)
    if not isinstance(checkpoint, Mapping) or weights not in checkpoint:
        raise ValueError(f"Source checkpoint is missing selected root weights key {weights!r}")
    if not isinstance(reference, Mapping) or "generator" not in reference:
        raise ValueError("Reference checkpoint is missing generator weights")
    state, mapping = renamed_weights(checkpoint[weights], reference["generator"], torch)
    destination.parent.mkdir(parents=True, exist_ok=True)
    metadata.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + f".{os.getpid()}.tmp")
    try:
        torch.save({weights: state}, temporary)
        saved = torch.load(temporary, map_location="cpu", mmap=True, weights_only=True)
        if set(saved) != {weights}:
            raise ValueError("Saved checkpoint root weights key changed")
        verify_tensors(state, saved[weights], torch)
        destination_hash = file_hash(temporary)
        result = dict(schema_version=SCHEMA, generated_at=datetime.now(timezone.utc).isoformat(),
                      weights=weights, reference_weights="generator", tensor_count=EXPECTED_COUNT,
                      source_prefix=SOURCE_PREFIX, destination_prefix=DESTINATION_PREFIX,
                      source=dict(path=str(source), sha256=source_hash, official_sha256=expected_source_hash,
                                  size_bytes=source.stat().st_size),
                      destination=dict(path=str(destination), sha256=destination_hash, size_bytes=temporary.stat().st_size),
                      reference=dict(path=str(reference_checkpoint), sha256=reference_hash,
                                     size_bytes=reference_checkpoint.stat().st_size, weights="generator"),
                      conversion="rename keys only; no tensor cast, clone, arithmetic, or value modification",
                      key_mapping=mapping,
                      validation=dict(strict_reference_keys=True, strict_reference_shapes=True,
                                      tensor_objects_preserved=True, reload_bitwise_equal=True,
                                      dtype_preserved=True, rename_only=True),
                      torch_version=torch.__version__, normalizer_sha256=file_hash(Path(__file__)))
        metadata_temporary = metadata.with_name(metadata.name + f".{os.getpid()}.tmp")
        metadata_temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+"\n")
        temporary.replace(destination)
        metadata_temporary.replace(metadata)
    finally:
        if temporary.exists():
            temporary.unlink()
    print(json.dumps(dict(status="normalized", destination=str(destination), metadata=str(metadata),
                          source_sha256=source_hash, destination_sha256=destination_hash, tensor_count=EXPECTED_COUNT)), flush=True)
    return result


def self_test():
    import torch

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        source, destination, reference, metadata = (root / name for name in ("raw.pt", "normalized.pt", "ref.pt", "normalization.json"))
        raw = {SOURCE_PREFIX+f"weight_{i}": torch.arange(i%5+1, dtype=torch.float32 if i%2 else torch.float16)
               for i in range(EXPECTED_COUNT)}
        ref = {DESTINATION_PREFIX+f"weight_{i}": torch.zeros_like(value) for i, value in enumerate(raw.values())}
        torch.save({"generator_ema": raw}, source)
        torch.save({"generator": ref}, reference)
        source_hash = file_hash(source)
        result = normalize(source, destination, "generator_ema", reference, metadata, source_hash)
        assert len(result["key_mapping"]) == EXPECTED_COUNT and file_hash(source) == source_hash
        before = (destination.stat().st_mtime_ns, metadata.stat().st_mtime_ns)
        normalize(source, destination, "generator_ema", reference, metadata, source_hash)
        assert before == (destination.stat().st_mtime_ns, metadata.stat().st_mtime_ns)
        bad = dict(raw)
        bad["unexpected.weight"] = bad.pop(next(iter(bad)))
        try:
            renamed_weights(bad, ref, torch)
        except ValueError:
            pass
        else:
            raise AssertionError("Unexpected source prefix must be rejected")
        bad = dict(ref)
        bad[next(iter(bad))] = torch.zeros(99)
        try:
            renamed_weights(raw, bad, torch)
        except ValueError:
            pass
        else:
            raise AssertionError("Reference shape mismatch must be rejected")
    print("PASS: 825-key rename, dtype/bitwise equality, raw preservation, verified resume without rewrite, invalid key/shape rejection")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--destination", type=Path)
    parser.add_argument("--weights", default="generator_ema")
    parser.add_argument("--reference-checkpoint", type=Path, default=Path("checkpoints/ode_init.pt"))
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--source-sha256")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)
    if args.self_test:
        self_test()
        return
    if any(getattr(args, name) is None for name in ("source", "destination", "metadata")):
        parser.error("--source, --destination and --metadata are required")
    normalize(args.source, args.destination, args.weights, args.reference_checkpoint,
              args.metadata, args.source_sha256)


if __name__ == "__main__":
    main()
