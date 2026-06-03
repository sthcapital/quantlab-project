\# quantlab



Quant research workspace for experiments, notebooks, backtests, and market data pipelines.



\## Setup



Create and activate the conda environment:



```powershell

conda activate quantlab

```



Install the package stack if needed:



```powershell

pip install numpy pandas scipy statsmodels scikit-learn pyarrow polars duckdb jupyterlab plotly sqlalchemy pydantic

```



\## Project structure



```text

quantlab-project/

|- src/

|  \\- quantlab/

|- tests/

|- data/

|  |- raw/

|  |- processed/

|  \\- external/

|- notebooks/

|- scripts/

|- output/

|- README.md

|- pyproject.toml

\\- .gitignore

```



\## Quick check



Run the environment validation script:



```powershell

python verify\_env.py

```



\## Notes



\- `data/raw/` is for untouched source data.

\- `data/processed/` is for cleaned or transformed datasets.

\- `output/` is for generated results, charts, and exports.

\- Keep reusable Python code in `src/quantlab/`.

