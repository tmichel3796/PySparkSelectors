"""Command-line entry point for PySparkSelectors."""

import typer

app = typer.Typer(help="PySpark column selector utilities.")


@app.callback()
def main() -> None:
    """Expose the PySparkSelectors command-line application."""


if __name__ == "__main__":
    app()
