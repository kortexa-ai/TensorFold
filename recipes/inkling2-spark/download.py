#!/usr/bin/env python3
"""Download one rank of the immutable public Inkling EXL3 pack; prepare native metadata separately."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import urllib.request

REPOSITORY = "kortexa-ai/Inkling-2-Small-EXL3-2.59bpw"
REVISION = "05c6cd0dff9df5bf20b10419c663a1e2adc7eb64"
MANIFEST_SHA256 = "97d75d66d3883514450772f8a3efb348467d35157925c908ea97315ae6e3feca"
URL = f"https://huggingface.co/{REPOSITORY}/resolve/{REVISION}/"
ASSETS = ("config.json", "hf_quant_config.json", "tokenizer.json", "tokenizer_config.json",
          "special_tokens_map.json", "chat_template.jinja")
# The published license files sit outside the tensor manifest; pin their exact bytes as well.
LICENSES = {
    "LICENSE": {"bytes": 11346, "sha256": "1495e1e757ef4d0925a2350563cf5754bb23c51701a8ec4fb3c5cdcbedae6747"},
    "NOTICE": {"bytes": 325, "sha256": "e71ff351f806c4f6ebd25f3c4ba9840ecd47283dc5ffcbd97829c59268369a47"},
}
DRAFT_FILES = {
    "auxiliary/dspark-fold.safetensors": {
        "revision": "f82bc6f03ec8c61a1d3afd155c820b71bf05bf7c",
        "bytes": 103326656,
        "sha256": "60971cba33c7344d48d75e3a7cc5fda6e5496ec6508389a57db0952b8fba5bb5",
    },
}


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(8 * 1024**2):
            h.update(chunk)
    return h.hexdigest()


def write(path, data):
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError(f"Existing file has different bytes: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".preparing")
    temp.write_bytes(data)
    temp.replace(path)


def prepare_metadata(root):
    config = json.loads((root / "auxiliary/config.json").read_text())
    config["quantization_config"] = {"quant_method": "exl3", "bits": 2.59375,
                                    "codebook": "mul1", "version": "1.6.0"}
    config["text_config"].pop("quantization_config", None)
    config["text_config"]["max_position_embeddings"] = 1048576
    # This text profile stops before the assistant message boundary enters the reply.
    config["eos_token_id"] = [200006, 200010]
    write(root / "config.json", (json.dumps(config, indent=2) + "\n").encode())
    for name in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"):
        write(root / name, (root / "auxiliary" / name).read_bytes())
    original = (root / "auxiliary/chat_template.jinja").read_text()
    # Numeric efforts already have numeric types; the native template engine has no float filter.
    original = original.replace("eff | float", "eff")
    framing = ('{%- if enable_thinking is not defined or not enable_thinking -%}'
               '{%- set reasoning_effort = "none" -%}{%- endif -%}')
    prefix = '{%- if add_generation_prompt -%}<|message_model|><|content_text|>{%- endif -%}'
    write(root / "chat_template.jinja", (framing + original + prefix).encode())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rank", type=int, choices=(0, 1), required=True)
    p.add_argument("--directory", type=Path, required=True)
    p.add_argument("--reuse", type=Path, action="append", default=[],
                   help="Optional existing copy of the same immutable pack; files are verified before reuse")
    p.add_argument("--offline", action="store_true")
    p.add_argument("--drafts", action="store_true", help="Also verify the bundled BF16 DSpark and folded head")
    args = p.parse_args()
    root = args.directory.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".download.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        owner = {"repository": REPOSITORY, "revision": REVISION, "rank": args.rank, "world": 2}
        write(root / "rank.json", (json.dumps(owner, indent=2) + "\n").encode())
        meta = root / "checkpoint.json"
        if not meta.exists():
            for src in args.reuse:
                candidate = src.expanduser() / "checkpoint.json"
                if candidate.is_file() and digest(candidate) == MANIFEST_SHA256:
                    shutil.copyfile(candidate, meta)
                    break
            else:
                if args.offline:
                    raise FileNotFoundError("The pinned checkpoint.json is unavailable offline")
                with urllib.request.urlopen(URL + "checkpoint.json", timeout=60) as response:
                    data = response.read(16 * 1024**2)
                if hashlib.sha256(data).hexdigest() != MANIFEST_SHA256:
                    raise ValueError("Checkpoint manifest hash mismatch")
                write(meta, data)
        if digest(meta) != MANIFEST_SHA256:
            raise ValueError("Existing checkpoint manifest differs from the pinned pack")
        checkpoint = json.loads(meta.read_text())
        files = {"dense/global.safetensors", *(f"auxiliary/{n}" for n in ASSETS), *LICENSES}
        for index in range(48):
            files.add(f"dense/layer-{index:02}.safetensors")
            for batch in checkpoint["layers"][str(index)]["batches"]:
                if args.rank * 192 <= batch["start"] < (args.rank + 1) * 192:
                    files.add(batch["file"])
        if args.drafts:
            files.update(("auxiliary/mtp.safetensors", *DRAFT_FILES))

        def fetch(name):
            if name.startswith("/") or ".." in Path(name).parts or "\\" in name:
                raise ValueError("Unsafe manifest path")
            target = root / name
            info = DRAFT_FILES.get(name) or LICENSES.get(name) or checkpoint["files"][name]
            sha, size = info["sha256"], info["bytes"]
            if target.exists():
                if target.stat().st_size != size or digest(target) != sha:
                    raise ValueError(f"Preserve the unexpected existing file and choose another directory: {target}")
                return
            target.parent.mkdir(parents=True, exist_ok=True)
            for src in args.reuse:
                candidate = src.expanduser() / name
                if candidate.is_file() and candidate.stat().st_size == size and digest(candidate) == sha:
                    try:
                        os.link(candidate, target)
                    except OSError:
                        shutil.copyfile(candidate, target)
                    return
            if args.offline:
                raise FileNotFoundError(name)
            temp = target.with_name(target.name + "." + sha[:12] + ".part")
            h = hashlib.sha256()
            url = f"https://huggingface.co/{REPOSITORY}/resolve/{info['revision']}/" if name in DRAFT_FILES else URL
            with urllib.request.urlopen(url + name, timeout=120) as source, temp.open("wb") as output:
                while block := source.read(8 * 1024**2):
                    h.update(block)
                    output.write(block)
            if temp.stat().st_size != size or h.hexdigest() != sha:
                raise ValueError(f"Downloaded bytes failed verification: {name}")
            temp.replace(target)

        with ThreadPoolExecutor(max_workers=4) as pool:
            for index, _ in enumerate(pool.map(fetch, sorted(files)), 1):
                if index % 100 == 0 or index == len(files):
                    print(f"rank {args.rank}: verified {index}/{len(files)} files", flush=True)
        prepare_metadata(root)
        print(json.dumps({**owner, "status": "verified", "directory": str(root),
                          "manifest_sha256": MANIFEST_SHA256, "files": len(files)}))


if __name__ == "__main__":
    main()
