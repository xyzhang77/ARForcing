"""Download or verify one pinned public model from an evaluation manifest."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import urllib.request


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def checksum(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download(partial, source, model_id):
    partial.parent.mkdir(parents=True, exist_ok=True)
    expected = source["size"]
    url = f'https://huggingface.co/{source["repo_id"]}/resolve/{source["revision"]}/{source["filename"]}?download=true'
    for attempt in range(5):
        offset = partial.stat().st_size if partial.exists() else 0
        if offset == expected:
            return
        if offset > expected:
            raise ValueError(f"Partial file larger than official size: {partial}")
        try:
            headers = {"Range": f"bytes={offset}-"} if offset else {}
            request = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(request, timeout=60) as response:
                if offset and (response.status != 206 or not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-")):
                    raise ValueError("Server did not return the requested byte range")
                reported = time.monotonic()
                with partial.open("ab" if offset else "wb") as stream:
                    while True:
                        block = response.read(8*1024*1024)
                        if not block:
                            break
                        stream.write(block)
                        offset += len(block)
                        if time.monotonic()-reported >= 10:
                            print(json.dumps(dict(model=model_id, downloaded=offset, total=expected,
                                                  fraction=round(offset/expected, 4))), flush=True)
                            reported = time.monotonic()
            if offset != expected:
                raise ValueError(f"Downloaded {offset}, expected {expected} bytes")
            return
        except (OSError, TimeoutError) as error:
            print(json.dumps(dict(model=model_id, retry=attempt+1, error=type(error).__name__)), flush=True)
            if attempt == 4:
                raise
            time.sleep(2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model", required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--download-only", action="store_true")
    mode.add_argument("--finalize", action="store_true")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    model = next(item for item in manifest["models"] if item["id"] == args.model)
    source = dict(model["download"])
    destination = Path(source.pop("destination", model["checkpoint"]))
    partial = destination.with_suffix(".pt.download.part")
    source_path = Path(source.pop("source_file"))
    ready_path = source_path.with_name("download_ready.json")
    if args.download_only:
        if not destination.is_file():
            download(partial, source, args.model)
        candidate = destination if destination.is_file() else partial
        if candidate.stat().st_size != source["size"]:
            raise ValueError("Downloaded byte length does not match official model")
        atomic_json(ready_path, dict(source, model=args.model, download_complete=True,
                                    verified=False, partial=str(candidate)))
        print(json.dumps(dict(model=args.model, download_complete=True, verified=False,
                              ready=str(ready_path))), flush=True)
        return
    candidate = destination if destination.is_file() else partial
    print(json.dumps(dict(model=args.model, verifying_sha256=str(candidate))), flush=True)
    if candidate.stat().st_size != source["size"] or checksum(candidate) != source["sha256"]:
        raise ValueError(f"Official checkpoint size/SHA-256 mismatch: {candidate}")
    if candidate != destination:
        candidate.replace(destination)
    source.update(model=args.model, weights=model["weights"], complete=True,
                  source=f'https://huggingface.co/{source["repo_id"]}/blob/{source["revision"]}/{source["filename"]}',
                  verified_at_unix=time.time())
    atomic_json(source_path, source)
    print(json.dumps(dict(checkpoint=str(destination.resolve()), **source)), flush=True)


if __name__ == "__main__":
    main()
