"""Забирает образ из реестра через рабочий прокси и складывает в tar для docker load.

Нужен потому, что собственный ретранслятор прокси в Docker Desktop не
форвардит запросы, хотя сам прокси со стороны Windows работает.

    py -3 pull_image.py library/caddy 2-alpine caddy:2-alpine out.tar
"""
import gzip
import json
import os
import shutil
import sys
import tarfile
import tempfile
import time
import urllib.request
from pathlib import Path

PROXY = os.environ.get("REG_PROXY", "http://127.0.0.1:10808")
REGISTRY = "https://registry-1.docker.io"

opener = urllib.request.build_opener(
    urllib.request.ProxyHandler({"http": PROXY, "https": PROXY})
)

ACCEPT = ", ".join([
    "application/vnd.oci.image.index.v1+json",
    "application/vnd.docker.distribution.manifest.list.v2+json",
    "application/vnd.oci.image.manifest.v1+json",
    "application/vnd.docker.distribution.manifest.v2+json",
])


def fetch(url, token=None, timeout=600):
    headers = {"Accept": ACCEPT}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    # Реестр иногда отвечает с большой задержкой на рукопожатии;
    # одна повторная попытка снимает почти все такие срывы.
    last = None
    for attempt in range(3):
        try:
            return opener.open(urllib.request.Request(url, headers=headers), timeout=timeout)
        except Exception as exc:
            last = exc
            print(f"    попытка {attempt + 1} не удалась: {type(exc).__name__}", flush=True)
            time.sleep(3)
    raise last


def get_token(repo):
    url = (f"https://auth.docker.io/token?service=registry.docker.io"
           f"&scope=repository:{repo}:pull")
    return json.load(fetch(url))["token"]


def download_blob(repo, digest, token, dest: Path, label: str) -> None:
    started = time.time()
    with fetch(f"{REGISTRY}/v2/{repo}/blobs/{digest}", token) as response:
        with dest.open("wb") as out:
            shutil.copyfileobj(response, out, length=1024 * 256)
    size = dest.stat().st_size
    seconds = max(time.time() - started, 0.01)
    print(f"  {label}: {size / 1e6:.1f} МБ за {seconds:.1f}с "
          f"({size / seconds / 1e6:.2f} МБ/с)", flush=True)


def main() -> int:
    repo, reference, tag, output = sys.argv[1:5]

    print(f"Образ {repo}:{reference} через прокси {PROXY}", flush=True)
    token = get_token(repo)

    index = json.load(fetch(f"{REGISTRY}/v2/{repo}/manifests/{reference}", token))
    if "manifests" in index:
        digest = next(
            m["digest"] for m in index["manifests"]
            if m.get("platform", {}).get("architecture") == "amd64"
            and m.get("platform", {}).get("os") == "linux"
        )
        manifest = json.load(fetch(f"{REGISTRY}/v2/{repo}/manifests/{digest}", token))
    else:
        manifest = index

    work = Path(tempfile.mkdtemp(prefix="pull-"))
    try:
        config_digest = manifest["config"]["digest"]
        config_name = config_digest.split(":")[1] + ".json"
        download_blob(repo, config_digest, token, work / config_name, "конфигурация")

        layer_paths = []
        for i, layer in enumerate(manifest["layers"], start=1):
            hexed = layer["digest"].split(":")[1]
            layer_dir = work / hexed
            layer_dir.mkdir()
            packed = work / f"{hexed}.gz"
            download_blob(repo, layer["digest"], token, packed,
                          f"слой {i}/{len(manifest['layers'])}")

            # docker load ожидает распакованные layer.tar
            target = layer_dir / "layer.tar"
            if layer["mediaType"].endswith("gzip"):
                with gzip.open(packed, "rb") as src, target.open("wb") as dst:
                    shutil.copyfileobj(src, dst, length=1024 * 256)
            else:
                packed.rename(target)
            packed.unlink(missing_ok=True)
            layer_paths.append(f"{hexed}/layer.tar")

        (work / "manifest.json").write_text(
            json.dumps([{
                "Config": config_name,
                "RepoTags": [tag],
                "Layers": layer_paths,
            }]),
            encoding="utf-8",
        )

        out_path = Path(output)
        with tarfile.open(out_path, "w") as tar:
            for item in sorted(work.iterdir()):
                tar.add(item, arcname=item.name)
        print(f"Готово: {out_path} ({out_path.stat().st_size / 1e6:.1f} МБ)")
        print(f"Загрузить: docker load -i {out_path}")
        return 0
    finally:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
