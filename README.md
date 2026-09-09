# PySparkSelectors

![PyPI version](https://img.shields.io/pypi/v/PySpark_Column_Selectors.svg)

Adds column selector functionality to pyspark

* GitHub: https://github.com/tmichel3796/PySpark_Column_Selectors/
* PyPI package: https://pypi.org/project/PySpark_Column_Selectors/
* Created by: **[Trevor A. Michel](trevormichel.com)** | GitHub https://github.com/tmichel3796 | PyPI https://pypi.org/user/tmichel3796/
* Free software: MIT License

## Features

* TODO

## Documentation

Documentation is built with [Zensical](https://zensical.org/) and deployed to GitHub Pages.

* **Live site:** https://tmichel3796.github.io/PySparkSelectors/
* **Preview locally:** `just docs-serve` (serves at http://localhost:8000)
* **Build:** `just docs-build`

API documentation is auto-generated from docstrings using [mkdocstrings](https://mkdocstrings.github.io/).

Docs deploy automatically on push to `main` via GitHub Actions. To enable this, go to your repo's Settings > Pages and set the source to **GitHub Actions**.

## Development

To set up for local development:

```bash
# Clone your fork
git clone git@github.com:your_username/PySpark_Column_Selectors.git
cd PySpark_Column_Selectors

# Install in editable mode with live updates
uv tool install --editable .
```

This installs the CLI globally but with live updates - any changes you make to the source code are immediately available when you run `PySparkSelectors`.

Run tests:

```bash
uv run pytest
```

Run quality checks (format, lint, type check, test):

```bash
just qa
```

## Author

PySparkSelectors was created in 2026 by Trevor A. Michel.

Built with [Cookiecutter](https://github.com/cookiecutter/cookiecutter) and the [audreyfeldroy/cookiecutter-pypackage](https://github.com/audreyfeldroy/cookiecutter-pypackage) project template.
