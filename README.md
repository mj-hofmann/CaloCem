![Logo](https://github.com/mj-hofmann/CaloCem/blob/main/icon/icon.png?raw=true)

# Loading and Processing of Data Files from TAM Air Calorimeters.

After collecting multiple experimental results files from a TAM Air calorimeter you will be left with multiple `.xls` or `.csv`-files obtained as exports from the device control software. **CaloCem** is here to make calorimetry data processing easy.

The package was written with a strong focus on the calorimetry of **cementitious materials**.

## Features

Some of the features are:

* Batch calorimetry data processing
* Plotting
* Tian correction
* Feature extraction (cumulative heat, time of dormant period, gradient of silicate reaction, etc.)

## Documentation
The full documentation is available [here](https://mj-hofmann.github.io/CaloCem/).

## Download Stats

[![PyPI - Downloads](https://img.shields.io/pypi/dm/calocem.svg?color=blue&label=Downloads&logo=pypi&logoColor=gold)](https://pepy.tech/project/calocem)
[![PyPI - Downloads](https://static.pepy.tech/personalized-badge/calocem?period=total&units=none&left_color=black&right_color=grey&left_text=Downloads)](https://pepy.tech/project/tainstcalorimetry)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/calocem.svg?logo=python&label=Python&logoColor=gold)](https://pypi.org/project/calocem/) 

## Quickstart

```bash
pip install calocem
```

```python
from pathlib import Path

import matplotlib.pyplot as plt

from calocem import Measurement, ProcessingParameters

# folder containing the exported .csv or .xls files
tam = Measurement(folder=Path("calo_data"), show_info=False)

tam.plot(t_unit="h")
plt.show()

processparams = ProcessingParameters()
processparams.cutoff.cutoff_min = 30  # ignore the first 30 minutes

peaks = tam.get_peaks(processparams, show_plot=False)
mainpeak = tam.get_mainpeak_params(processparams)
```

See the [documentation](https://mj-hofmann.github.io/CaloCem/) for loading
metadata, Tian correction, slope and onset analysis and more.

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install CaloCem. Python 3.10 or newer is required.

```bash
pip install CaloCem
```

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

List of contributors:
- [mj-hofmann](https://github.com/mj-hofmann)
- [tgaedt](https://github.com/tgaedt)

## License
[GNU GPLv3](https://choosealicense.com/licenses/gpl-3.0/#)


## Test
![Tests](https://github.com/mj-hofmann/CaloCem/actions/workflows/run-tests.yml/badge.svg)

## Code Style

Code is checked with [ruff](https://docs.astral.sh/ruff/) and [mypy](https://mypy-lang.org/):

```bash
uv run ruff check calocem tests
uv run mypy
```
