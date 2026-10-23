# CaloCem

CaloCem reads, plots and analyses isothermal heat flow calorimetry data,
with a focus on the hydration of cementitious materials. It loads the
exports of TAM Air calorimeters, combines many experiments in one
[pandas](https://pandas.pydata.org/) DataFrame and extracts characteristic
values such as cumulative heat, peak positions, the dormant period and the
slope of the main hydration peak.

!!! warning

    CaloCem has been developed without involvement of *TA Instruments* and is
    thus independent from the company and its software.

## Installation

CaloCem requires Python 3.10 or newer and is available on
[PyPI](https://pypi.org/project/calocem/):

```bash
pip install calocem
```

## Quickstart

Put the exported `.csv` or `.xls` files of your experiments into one folder,
e.g. `calo_data`, next to your script:

```bash
.
├── myscript.py
└── calo_data
    ├── opc_1.csv
    └── opc_2.csv
```

Load and plot the data:

```python
from pathlib import Path

import matplotlib.pyplot as plt

from calocem import Measurement, ProcessingParameters

datapath = Path(__file__).parent / "calo_data"
tam = Measurement(folder=datapath, show_info=False)

ax = tam.plot(t_unit="h")
ax.set_xlim(0, 48)
plt.show()
```

The data of all samples is available as a DataFrame:

```python
data = tam.get_data()
print(tam.get_sample_names())
```

Analysis settings are collected in `ProcessingParameters` and passed to the
analysis methods, which return DataFrames with one row per sample:

```python
processparams = ProcessingParameters()
processparams.cutoff.cutoff_min = 30  # ignore the first 30 minutes

peaks = tam.get_peaks(processparams, show_plot=False)
mainpeak = tam.get_mainpeak_params(processparams)

print(mainpeak[["sample_short", "gradient_from_max_slope", "onset_time_min_max_slope"]])
```

## Where to go next

* [Loading Data](loading.md): supported files, metadata, averaging,
  downsampling, saving and reloading.
* [Processing Parameters](processing-parameters.md): all analysis settings
  and what they affect.
* [Plotting](plotting_example.ipynb): customizing plots, subplots, plotting by
  metadata category.
* [Characteristic Values](quantification.md): cumulative heat, peaks, dormant
  period, ASTM C1679, deconvolution.
* [Main Peak Parameters](example_get_mainpeak_params.ipynb): slopes and onset
  of the main hydration peak.
* [Tian Correction](tian.md): correcting the thermal inertia of the
  calorimeter.
* [Reference](reference/measurement.md): all methods and parameters.

More scripts are available in the
[examples folder](https://github.com/mj-hofmann/CaloCem/tree/main/examples)
of the repository.
