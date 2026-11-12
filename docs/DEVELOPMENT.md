# Development

The project uses a regular `lcgcn` package with relative imports, a `pyproject.toml` build definition, and a console entry point. `main.py` only delegates to `lcgcn.cli.main`. No `sys.path` injection or editor warning suppression is used. Pyright checks import resolution; full static typing is not claimed.

Install development tools with `python -m pip install -e ".[dev]"`. Run `pyright` and `python -m unittest discover -s tests -v`. Synthetic tests require no downloaded datasets. The GitHub Actions workflow runs these checks and configuration listing on Windows and Ubuntu with Python 3.10. See the repository Actions tab for current results; local validation alone is not evidence of a remote CI pass.

The code and documentation use the MIT license. No unverified paper DOI, bibliographic entry, or personal author identity is supplied. Dataset rights remain unresolved (see DATA.md); local data are excluded by `.gitignore` and are not included in wheels. Do not upload the complete local data archives unless their redistribution rights are established.

Contributions should describe the problem, change, and relevant validation. For numerical changes, report graph construction, training protocol, split/seed treatment, and any score differences; do not select runs by test accuracy. Preserve the existing frozen recipes when adding new protocols.
