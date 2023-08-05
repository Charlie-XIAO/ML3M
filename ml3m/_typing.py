"""This file is intended for convenient typing."""

from typing import Literal

import pandas as pd


DataItemType = pd.Series | list | dict
DatasetFormat = Literal["jsonl", "json", "csv"]
