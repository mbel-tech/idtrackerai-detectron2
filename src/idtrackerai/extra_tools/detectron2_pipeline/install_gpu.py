"""Installs PyTorch and Detectron2 for the machine it is run on.

Neither can be an ordinary dependency, for reasons that are worth stating
because they look like oversights:

* PyPI's ``torch`` wheel for Windows is CPU-only (124 MB against roughly 2.5 GB
  for a CUDA build), because the CUDA builds live on PyTorch's own index.
  Declaring ``torch`` would therefore install a CPU-only PyTorch over whatever
  the user had, and idtracker.ai's *own* identity tracking would quietly stop
  using the GPU. Upstream leaves it out for exactly this reason and documents
  the pytorch.org command instead.
* Detectron2 is not on PyPI at all. It is built from source, which needs a C++
  compiler and a CUDA toolkit matching the installed torch, so declaring it
  would make ``pip install`` fail outright on every machine without a
  compiler -- including machines that only ever wanted to track.

So the install is a command rather than metadata. It works out what this
machine needs, shows exactly what it will run, and only then runs it.
"""

import json
import platform
import re
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

DETECTRON2_URL = "git+https://github.com/facebookresearch/detectron2.git"

# PyTorch publishes one index per CUDA version. Newest first: the highest that
# the installed driver supports is the one to use.
CUDA_INDEXES = ("cu128", "cu126", "cu124", "cu121", "cu118")

INDEX_ROOT = "https://download.pytorch.org/whl/"


@dataclass
class Machine:
    """What this computer can offer PyTorch."""

    kind: str = "cpu"  # nvidia | rocm | apple | cpu
    gpu_name: str | None = None
    cuda_version: tuple[int, int] | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def description(self) -> str:
        if self.kind == "nvidia":
            cuda = (
                f", CUDA {self.cuda_version[0]}.{self.cuda_version[1]}"
                if self.cuda_version
                else ""
            )
            return f"NVIDIA GPU: {self.gpu_name or 'present'}{cuda}"
        if self.kind == "apple":
            return f"Apple {self.gpu_name or 'M-series'} (Metal)"
        if self.kind == "rocm":
            return f"AMD GPU: {self.gpu_name or 'present'} (ROCm)"
        return "No GPU that PyTorch can use; a CPU build will be installed"


def _run(command: list[str], timeout: float = 30.0) -> str:
    try:
        done = subprocess.run(
            command, capture_output=True, text=True, timeout=timeout
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return done.stdout or ""


def detect_machine() -> Machine:
    """What GPU is here, if any. Never raises."""
    system = platform.system()

    if shutil.which("nvidia-smi"):
        name = _run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"]
        ).strip().splitlines()
        banner = _run(["nvidia-smi"])
        found = re.search(r"CUDA Version:\s*(\d+)\.(\d+)", banner)
        if name or found:
            return Machine(
                kind="nvidia",
                gpu_name=name[0].strip() if name else None,
                cuda_version=(
                    (int(found.group(1)), int(found.group(2))) if found else None
                ),
            )

    if system == "Darwin" and platform.machine() in ("arm64", "aarch64"):
        return Machine(kind="apple", gpu_name=platform.machine())

    if system == "Linux" and shutil.which("rocm-smi"):
        return Machine(kind="rocm")

    return Machine(kind="cpu")


def _index_exists(suffix: str, timeout: float = 10.0) -> bool:
    request = urllib.request.Request(INDEX_ROOT + suffix, method="HEAD")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return 200 <= response.status < 400
    except (urllib.error.URLError, OSError, ValueError):
        return False


def choose_index(machine: Machine, check: bool = True) -> str | None:
    """The ``--index-url`` for this machine, or None for plain PyPI.

    Apple silicon takes the default wheel, which already uses Metal. A CPU
    machine takes the default too, which is what PyPI serves.
    """
    if machine.kind != "nvidia":
        return None

    candidates = list(CUDA_INDEXES)
    if machine.cuda_version:
        driver = machine.cuda_version[0] * 10 + machine.cuda_version[1]
        # never ask for a CUDA newer than the driver supports
        candidates = [
            suffix
            for suffix in CUDA_INDEXES
            if int(suffix[2:]) <= driver
        ] or [CUDA_INDEXES[-1]]

    for suffix in candidates:
        if not check or _index_exists(suffix):
            return INDEX_ROOT + suffix
    return INDEX_ROOT + candidates[-1]


@dataclass
class Plan:
    """The commands that would be run, and why."""

    machine: Machine
    commands: list[list[str]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def as_text(self) -> str:
        lines = [f"This machine: {self.machine.description}", ""]
        lines += [
            "  " + " ".join(f'"{p}"' if " " in p else p for p in command)
            for command in self.commands
        ]
        if self.notes:
            lines += [""] + self.notes
        return "\n".join(lines)


def make_plan(python: str | None = None, check_index: bool = True) -> Plan:
    """What to install here, without installing anything."""
    python = python or sys.executable
    machine = detect_machine()
    plan = Plan(machine=machine)

    torch_command = [python, "-m", "pip", "install", "torch", "torchvision"]
    index = choose_index(machine, check=check_index)
    if index:
        torch_command += ["--index-url", index]
    plan.commands.append(torch_command)

    plan.commands.append(
        [python, "-m", "pip", "install", DETECTRON2_URL]
    )

    if machine.kind == "cpu":
        plan.notes.append(
            "Without a GPU, training and the contour export will be far too "
            "slow to be useful. The Colab notebook is the better route; this "
            "install only makes the code importable."
        )
    if machine.kind in ("apple", "rocm"):
        plan.notes.append(
            "Detectron2's CUDA kernels do not build on this platform. Some of "
            "it will work on CPU; training will not be quick."
        )
    if sys.version_info >= (3, 12):
        plan.notes.append(
            f"Detectron2 is built from source and was last released in 2021. "
            f"It declares python_requires>=3.7 with no upper bound, but it is "
            f"not tested on Python {sys.version_info.major}."
            f"{sys.version_info.minor} and the build may fail. If it does, a "
            "Python 3.11 environment is the well-trodden path."
        )
    plan.notes.append(
        "Building Detectron2 needs a C++ compiler: Visual Studio Build Tools "
        "on Windows, gcc/g++ elsewhere."
    )
    plan.notes.append("Restart the Segmentation App afterwards.")
    return plan


def run_plan(plan: Plan, echo=print) -> int:
    """Runs the commands in order, stopping at the first failure."""
    for command in plan.commands:
        echo("$ " + " ".join(command))
        try:
            completed = subprocess.run(command)
        except (OSError, subprocess.SubprocessError) as exc:
            echo(f"Could not run it: {exc}")
            return 1
        if completed.returncode != 0:
            echo(f"That step failed with exit code {completed.returncode}.")
            return completed.returncode
    return 0


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="idtrackerai_d2_install_gpu",
        description=(
            "Install PyTorch and Detectron2 for this machine, so the training "
            "and contour-export steps can run here instead of on Colab"
        ),
    )
    parser.add_argument(
        "--yes", action="store_true",
        help="install without asking; otherwise the plan is shown first",
    )
    parser.add_argument(
        "--show", action="store_true",
        help="print the plan as JSON and exit, installing nothing",
    )
    parser.add_argument(
        "--python", help="install into this interpreter instead of the current one"
    )
    parser.add_argument(
        "--no-index-check", action="store_true",
        help="skip verifying the PyTorch index exists (for offline use)",
    )
    args = parser.parse_args()

    plan = make_plan(args.python, check_index=not args.no_index_check)

    if args.show:
        print(json.dumps(
            {
                "machine": plan.machine.description,
                "kind": plan.machine.kind,
                "commands": plan.commands,
                "notes": plan.notes,
            },
            indent=2,
        ))
        return 0

    print(plan.as_text())
    print()
    if not args.yes:
        try:
            answer = input("Run these now? [y/N] ").strip().lower()
        except EOFError:
            answer = ""
        if answer not in ("y", "yes"):
            print("Nothing was installed.")
            return 0

    code = run_plan(plan)
    if code == 0:
        print("\nChecking what the result can do...")
        from .gpu import describe_gpu

        print(describe_gpu(args.python).summary())
    return code


if __name__ == "__main__":
    raise SystemExit(main())
