"""
MUD CLI

CLI for MUD library
"""
import json
from typing import List

import click
import matplotlib.pyplot as plt
import numpy as np

from mud.examples.examples import examples

__author__ = "Carlos del-Castillo-Negrete"
__copyright__ = "Carlos del-Castillo-Negrete"
__license__ = "mit"

@click.group()
def cli():
    pass

cli.add_command(examples)

