"""
MUD CLI

CLI for MUD library
"""
import click

from mud.examples.examples import examples


@click.group()
def cli():
    pass


cli.add_command(examples)
