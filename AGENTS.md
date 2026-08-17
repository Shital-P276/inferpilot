# Agent Instructions — InferPilot

## Python environment

**Use the venv at `~/venv_inferpilot`** (native Linux filesystem, outside the project folder) for all Python commands — running scripts, installing packages, anything that needs Python.

```bash
source ~/venv_inferpilot/bin/activate
```

**Do not create or use a venv inside the project folder** (e.g. `venv_linux/`, `.venv/`, etc.) — a venv on `/mnt/d/` (this project's location) writes many small files during creation/installs, which hangs or becomes extremely slow due to how WSL bridges Windows-mounted drives. This already happened once and is a known, confirmed issue for this project, not a hypothetical.

If `~/venv_inferpilot` doesn't exist yet, ask before creating one — don't create a new venv inside the project directory as a fallback.

## Project structure notes

- This project also has a separate **Windows-native venv** at `venv/` inside the project folder — that one is for the user's manual/native-Windows workflow and Docker. Don't touch, modify, or install into it. It's unrelated to the WSL/Linux venv above.
- `requirements_wsl.txt` (generated from `requirements.txt` with `pywin32` removed, which is Windows-only and doesn't install on Linux) is the correct requirements file to use inside `~/venv_inferpilot`. Don't use `requirements.txt` directly here.
- `requirements_docker.txt` is for the Docker build specifically (different torch/CUDA version, intentional) — not related to either venv above.

## Paths

- Path values read from `router/utility_labels.csv` (and similar files) use Windows-style backslashes (`data\test\...`). On Linux these must be normalized to forward slashes before use (e.g. `.replace("\\", "/")`) or file loading will silently fail. This fix has been applied inside the shared `CorrectnessGateDataset` class — don't re-introduce backslash-path bugs in new code that reads this CSV.
