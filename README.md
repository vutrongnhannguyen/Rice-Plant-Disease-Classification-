# Getting Started

This project is managed by [uv](https://docs.astral.sh/uv/) to correctly resolve Python and its dependencies.

To install the project dependencies, run

```bash
uv sync
```

Additionally, activate the Python virtual environment

```bash
uv venv
source .venv/bin/activate
```

Start the Jupyter Notebook environment with

```bash
uv run --with jupyter jupyter lab
```

To start the project web server for getting predictions, run

```bash
uv run webapp.py
```

Here, uv ensure every run scripts has it dependencies resolve correctly and environment are self-contained.

## References

[Using uv with Jupyter](https://docs.astral.sh/uv/guides/integration/jupyter/)
