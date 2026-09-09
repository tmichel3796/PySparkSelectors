"""Console script for PySparkSelectors."""

import typer
from rich.console import Console

from PySparkSelectors import utils

app = typer.Typer()
console = Console()


@app.command()
def main() -> None:
    """Console script for PySparkSelectors."""
    console.print("Replace this message by putting your code into PySparkSelectors.cli.main")
    console.print("See Typer documentation at https://typer.tiangolo.com/")
    utils.do_something_useful()


if __name__ == "__main__":
    app()
