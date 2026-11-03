"""Source-file filtering rules, shared between repo_mapper and the v3 Code Explorer.

These were originally inline in repo_mapper. Pulled out when the Code Explorer
landed so both agents can stay in sync — adding a new vendored-deps directory
or a new source extension shouldn't require updating two files.
"""
from __future__ import annotations

# Path fragments that almost never contain useful signal for issue triage —
# lockfiles, vendored deps, build artifacts, minified output.
IGNORE_FRAGMENTS: tuple[str, ...] = (
    "/node_modules/", "/dist/", "/build/", "/.git/", "/.next/", "/.venv/",
    "/__pycache__/", "/vendor/", "/target/", "/out/",
    "package-lock.json", "yarn.lock", "pnpm-lock.yaml", "poetry.lock",
    "uv.lock", "cargo.lock", ".min.js", ".min.css",
)

# Extensions worth handing to an agent. Adding a new source language?
# It probably goes here.
SOURCE_EXTENSIONS: frozenset[str] = frozenset({
    ".py", ".pyi", ".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs",
    ".go", ".rs", ".java", ".kt", ".swift", ".rb", ".php",
    ".c", ".h", ".cc", ".cpp", ".hpp", ".cs",
    ".vue", ".svelte", ".astro",
    ".sh", ".bash", ".ps1",
    ".sql", ".graphql", ".proto",
    ".md", ".mdx",
    ".toml", ".yaml", ".yml", ".json",
})

# Extensionless filenames we want to keep (Makefiles, Dockerfiles, etc.).
_EXTENSIONLESS_KEEP: frozenset[str] = frozenset({
    "Dockerfile", "Makefile", "Procfile", "go.mod", "Containerfile",
})

# Skip giant files — LLMs choke on them, and they're rarely the right
# files to edit. (Generated SQL dumps, megabyte-sized fixtures, etc.)
MAX_BLOB_SIZE_BYTES = 200_000


def is_interesting_path(path: str) -> bool:
    """True if this is the kind of file an agent should consider."""
    lowered = "/" + path.lower()
    if any(frag in lowered for frag in IGNORE_FRAGMENTS):
        return False
    if "." in path:
        ext = "." + path.rsplit(".", 1)[1].lower()
        if ext in SOURCE_EXTENSIONS:
            return True
    return path.rsplit("/", 1)[-1] in _EXTENSIONLESS_KEEP
