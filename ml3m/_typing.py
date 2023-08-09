"""This file is for convenient typing."""

from __future__ import annotations

from typing import Literal

import pandas as pd

AggregateMethod = Literal["mean", "sum", "min", "max", "mode"]
DataItemType = pd.Series | list | dict
DatasetFormat = Literal["jsonl", "json", "csv"]
LoggingMode = Literal["all", "failed", "none"]
QaSubject = Literal["accuracy", "completeness", "clarity"]
