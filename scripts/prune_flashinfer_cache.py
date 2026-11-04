"""Delete flashinfer_jit_cache modules that cannot run on any of the given GPU architectures.

Usage: prune_flashinfer_cache.py ARCH...   (e.g. 8.6 8.9; "all" keeps everything)

Each cached module embeds its device code in CUDA fat binaries that list a target architecture per entry. A module
can run on sm_XY if it has SASS for the same major version at or below XY, or PTX at or below XY (the driver
JIT-compiles PTX). A module none of the given architectures can run is deleted: on those GPUs it would only fail at
launch, or fall back to JIT compilation, which needs nvcc. Whole modules are kept or deleted; binaries are not edited.
Anything the parser does not recognize (no fat binary found, unknown entry kinds) is kept.
"""

import os
import shutil
import struct
import sys

import flashinfer_jit_cache

MAGIC = struct.pack("<I", 0xBA55ED50)
PTX, SASS = 1, 2


def entries(blob):
    """(kind, arch) for every fat-binary entry in an ELF file's bytes."""
    start = blob.find(MAGIC)
    while start != -1:
        _, _, header_size, fat_size = struct.unpack_from("<IHHQ", blob, start)
        pos, end = start + header_size, start + header_size + fat_size
        while pos + 32 <= end:
            kind, _, entry_header, size = struct.unpack_from("<HHIQ", blob, pos)
            if kind not in (PTX, SASS) or entry_header == 0:
                break
            yield kind, struct.unpack_from("<I", blob, pos + 28)[0]
            pos += entry_header + size
        start = blob.find(MAGIC, max(end, start + 4))


def runs_on(found, arch):
    return any(
        (kind == SASS and code // 10 == arch // 10 and code <= arch) or (kind == PTX and code <= arch)
        for kind, code in found
    )


def module_entries(path):
    found = set()
    for dirpath, _, names in os.walk(path):
        for name in names:
            if name.endswith(".so"):
                with open(os.path.join(dirpath, name), "rb") as f:
                    found |= set(entries(f.read()))
    return found


def size(path):
    return sum(os.path.getsize(os.path.join(d, n)) for d, _, names in os.walk(path) for n in names)


def main(args):
    if not args or "all" in args:
        print("FlashInfer JIT cache: keeping every module")
        return
    archs = [int(round(float(a) * 10)) for a in args]
    root = os.path.join(os.path.dirname(flashinfer_jit_cache.__file__), "jit_cache")
    kept = removed = freed = 0
    for module in sorted(os.listdir(root)):
        path = os.path.join(root, module)
        found = module_entries(path)
        if not found or any(runs_on(found, arch) for arch in archs):
            kept += 1
            continue
        freed += size(path)
        shutil.rmtree(path)
        removed += 1
    print(f"FlashInfer JIT cache: kept {kept} modules, removed {removed} unusable on sm_{','.join(map(str, archs))} "
          f"({freed / 1e9:.2f} GB)")
    if kept == 0:
        sys.exit("refusing to leave an empty JIT cache")


if __name__ == "__main__":
    main(sys.argv[1:])
