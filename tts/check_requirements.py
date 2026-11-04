"""Check that the given requirements and all their dependencies are satisfied for this interpreter.

Stands in for `pip check` in the TTS virtualenv: that venv inherits the system packages, so `pip check` there also
reports vLLM's transformers>=5 requirement against the venv's transformers 4.57.3, which is the point of the venv.
Distributions resolve in import order, so a venv package shadows the system one, as it does at runtime.
"""

import sys
from importlib.metadata import PackageNotFoundError, distribution

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name


def problems(roots):
    found, seen, stack = [], set(), [(Requirement(r), "requested") for r in roots]
    while stack:
        req, parent = stack.pop()
        try:
            dist = distribution(req.name)
        except PackageNotFoundError:
            found.append(f"{parent} requires {req}, which is not installed")
            continue
        if not req.specifier.contains(dist.version, prereleases=True):
            found.append(f"{parent} requires {req}, but {dist.version} is installed")
            continue
        extras = tuple(sorted(req.extras))
        if (canonicalize_name(req.name), extras) in seen:
            continue
        seen.add((canonicalize_name(req.name), extras))
        for line in dist.requires or []:
            sub = Requirement(line)
            if sub.marker is None or any(sub.marker.evaluate({"extra": e}) for e in extras or ("",)):
                stack.append((Requirement(str(sub).split(";")[0]), f"{dist.metadata['Name']} {dist.version}"))
    return found


if __name__ == "__main__":
    issues = problems(sys.argv[1:])
    print("\n".join(issues) or f"Requirements satisfied: {', '.join(sys.argv[1:])}")
    sys.exit(1 if issues else 0)
