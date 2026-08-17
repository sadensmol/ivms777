"""Enforcement gate for "one model process, never twice" (CLAUDE.md, design
§5.1, plan 15 task 9). `app`/`worker`/`ingest`/`chat`/`search`/`web` are thin
HTTP clients of the `models` service — they must NEVER import torch,
transformers, or bitsandbytes (directly or transitively), never import
`embedding.siglip` (the real SigLIP lives only in the service), and never
construct `inference.client.OpenAICompatClient` or read
`Settings.inference_base_url` (only `modelsvc` may talk to Ollama directly).

Two complementary checks, both required — each catches what the other misses:

- The RUNTIME gate proves the actual import graph, as executed, never pulls a
  heavy lib into `sys.modules` — including via a lazy import that fires at
  module load time from somewhere deep in the call graph.
- The STATIC gate proves it by reading source, so it also catches a forbidden
  import sitting in a branch/try-except that today's runtime gate happens not
  to execute (e.g. an `except ImportError: import torch` fallback nobody hit
  yet). A future violation trips whichever check reaches it first — most will
  trip both.
"""
import ast
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Files that may import torch/transformers/bitsandbytes and build the Ollama
# client directly — the `models` service side of the codebase (design §5.1).
# `modelsvc/` itself is never scanned at all: only the dirs listed in
# `_thin_client_files` below are walked.
_EXEMPT_FILES = {
    REPO_ROOT / "embedding" / "siglip.py",
    REPO_ROOT / "embedding" / "text_embedder.py",  # caption-text embedder, models-service-only
    REPO_ROOT / "inference" / "client.py",
}

_FORBIDDEN_TOP_MODULES = {"torch", "transformers", "bitsandbytes"}


def _thin_client_files() -> list[Path]:
    files = [REPO_ROOT / "config.py"]
    for d in ("chat", "search", "ingest", "web", "albums", "storage", "db", "embedding", "inference"):
        files += sorted((REPO_ROOT / d).rglob("*.py"))
    return [p for p in files if p not in _EXEMPT_FILES and "__pycache__" not in p.parts]


def _violations_in_file(path: Path) -> list[str]:
    rel = path.relative_to(REPO_ROOT)
    tree = ast.parse(path.read_text(), filename=str(path))
    violations = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                top = alias.name.split(".")[0]
                if top in _FORBIDDEN_TOP_MODULES:
                    violations.append(
                        f"{rel}:{node.lineno}: `import {alias.name}` — thin-client code must not import {top}"
                    )
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            top = mod.split(".")[0]
            if top in _FORBIDDEN_TOP_MODULES:
                violations.append(
                    f"{rel}:{node.lineno}: `from {mod} import ...` — thin-client code must not import {top}"
                )
            elif mod == "embedding.siglip":
                violations.append(
                    f"{rel}:{node.lineno}: `from embedding.siglip import ...` — SigLIP lives only in the"
                    " models service, never in a thin client"
                )
        elif isinstance(node, ast.Call):
            fn = node.func
            name = fn.id if isinstance(fn, ast.Name) else getattr(fn, "attr", None)
            if name == "OpenAICompatClient":
                violations.append(
                    f"{rel}:{node.lineno}: constructs `OpenAICompatClient(...)` — only the models"
                    " service may build the Ollama client"
                )
        elif isinstance(node, ast.Attribute) and node.attr == "inference_base_url":
            violations.append(
                f"{rel}:{node.lineno}: accesses `.inference_base_url` — only the models service"
                " reads it and talks to Ollama directly"
            )
    return violations


def test_thin_client_source_never_imports_heavy_libs_or_builds_ollama_client():
    # 1b — static AST gate (design §5.1, CLAUDE.md "One model process"): walk
    # every .py under the thin-client set and assert none of them imports
    # torch/transformers/bitsandbytes/embedding.siglip, or constructs the
    # Ollama client / reads its base url directly (Task-4 parked minor).
    violations = []
    for path in _thin_client_files():
        violations += _violations_in_file(path)
    assert not violations, "one-model-process violation(s):\n" + "\n".join(violations)


def test_thin_client_entrypoints_never_load_torch_transformers_bitsandbytes():
    # 1a — runtime gate, in a FRESH subprocess (mirrors tests/test_coordinator_noop.py):
    # pytest's own collection may import tests that legitimately pull torch
    # (test_siglip_release.py, etc.), which would pollute sys.modules for an
    # in-process check. Importing the actual thin-client entrypoints here
    # proves no lazy import fires a heavy lib at module load time.
    script = (
        "import config, web.deps, web.app, ingest.cli, ingest.pipeline\n"
        "import chat.agent, chat.retrieve, search.retriever, search.semantic\n"
        "import sys\n"
        "loaded = sys.modules.keys() & {'torch', 'transformers', 'bitsandbytes'}\n"
        "assert not loaded, f'heavy lib(s) pulled in via thin-client import graph: {sorted(loaded)}'\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, cwd=REPO_ROOT, check=False,
    )
    assert result.returncode == 0, result.stderr
