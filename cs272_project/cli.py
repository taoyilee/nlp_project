"""Console script for cs272_project."""
import sys
import click


@click.command()
def main(args=None):
    """Console script for cs272_project."""
    click.echo("Replace this message by putting your code into "
               "cs272_project.cli.main")
    click.echo("See click documentation at https://click.palletsprojects.com/")
    return 0


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
