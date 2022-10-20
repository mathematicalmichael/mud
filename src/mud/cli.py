"""
MUD CLI

CLI for MUD library
"""
import click
import json
import numpy as np
from typing import List
import matplotlib.pyplot as plt

from mud.examples.examples import examples

__author__ = "Carlos del-Castillo-Negrete"
__copyright__ = "Carlos del-Castillo-Negrete"
__license__ = "mit"

@click.group()
def cli():
    pass

cli.add_command(examples)

