# Prerequisites

- Python 3.11
- Keras and Tensorflow 2.15.0
- Dependencies specified in `pyproject.toml`

# Getting Started

This project is managed by the [uv package and project manager](https://docs.astral.sh/uv/) to correctly resolve Python and its dependencies.

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

To generate a prediction file with test images, put corresponding model in the project folder and run

```bash
uv run predict.py
```

To start the project web server for getting predictions, run

```bash
uv run webapp.py
```

Here, uv ensure every run scripts has it dependencies resolve correctly and environment are self-contained.

# Entrypoints

| Priority | Notebook/Script                    | Description                                                                |
|----------|------------------------------------|----------------------------------------------------------------------------|
| 1        | `Dataset/separate.py`              | Entrypoint to generate additional data for training all 3 tasks            |
| 2        | `eda.ipynb`                        | Entrypoint to generate analyze and clean metadata for training all 3 tasks |
| 3        | `task1_label_classification.ipynb` | Task 1 training and evaluation                                             |
| 4        | `task2_4colors.ipynb`              | Task 2 training and evaluation                                             |
| 5        | `task3_3colors.ipynb`              | Task 3 training and evaluation                                             |
| 6        | `predict.py`                       | Predict test data using exported models                                    |
| 7        | `webapp.py`                        | API endpoints for getting on-the-fly predictions                           |

# Web Application

This project packages a simple single-page application (SPA) to use with the 3 models, located in `app/`.

With that, the application is bundled with [Vite](https://vite.dev/) and managed using `pnpm`.

To build the web application for production, run

```bash
cd app
pnpm i
pnpm build
```

Then, run the [Flask](https://flask.palletsprojects.com/en/stable/) web server to serve the both API endpoint and previously built SPA.

```bash
cd ..
uv run webapp.py
```

# References

[Using uv with Jupyter](https://docs.astral.sh/uv/guides/integration/jupyter/)
