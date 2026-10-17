# Loading Data

## Loading a folder

A `Measurement` reads all supported files of one folder (not its subfolders):

```python
from pathlib import Path
from calocem import Measurement

datapath = Path(__file__).parent / "calo_data"

tam = Measurement(folder=datapath, show_info=True)
```

With `show_info=True` the name of each file is printed while it is read.
Files that cannot be read are skipped with a warning.

To load only some files, pass a regular expression. It is matched against
the beginning of the file name:

```python
tam = Measurement(folder=datapath, regex=r"opc_.*\.csv")
```

## Supported files

| Export | Extension | Notes |
|---|---|---|
| TAM Air CSV export | `.csv` | The usual comma-separated export. A *Reaction start* time is used as time zero; periods between *Experiment interrupted* and *Experiment restarted* are removed. |
| TAM Air Excel export | `.xls` | Requires a sheet named *Raw data*; sample information is read from *SampleData*. `.xlsx` files are not loaded. |
| First-generation tab-separated export | `.csv` | Heat flow in mW with decimal commas; the sample mass is read from the file and the heat is integrated. |
| Data previously processed by CaloCem | `.csv` | Load with `processed=True`, e.g. a file written with `tam.get_data().to_csv(...)`. |

## The data

`get_data()` returns all samples in one
[pandas](https://pandas.pydata.org/) DataFrame in long format, i.e. one row
per sample and time:

```python
data = tam.get_data()
```

| Column | Unit | Description |
|---|---|---|
| `time_s` | s | Time since the start of the reaction. |
| `temperature_c` | °C | Temperature (not available for all exports). |
| `heat_flow_w` | W | Measured heat flow. |
| `heat_j` | J | Cumulated heat. |
| `normalized_heat_flow_w_g` | W/g | Heat flow per gram of sample. |
| `normalized_heat_j_g` | J/g | Cumulated heat per gram of sample. |
| `sample` | | Full path of the source file. |
| `sample_short` | | File name without extension, used as sample name. |

`get_sample_names()` lists the sample names, `get_information()` returns the
sample information stored in the files (e.g. the sample mass).

Some exports contain differently named temperature columns. Pass
`auto_clean=True` to merge them into `temperature_c` and to drop rows without
heat flow values.

If the mass stored in a file is wrong, the normalized columns can be
recalculated for one sample:

```python
tam.normalize_sample_to_mass("opc_1", mass_g=6.05)
```

## Adding metadata

Information about the samples, e.g. cement type or water-to-cement ratio, is
usually kept in a separate table. Add it as CSV or Excel file with one row per
sample and a column that contains the sample names (the file names without
extension):

```text
experiment_nr,date,cement_name,cement_amount_g,water_amount_g
calorimetry_data_1,01.01.2023,cem_a,100,50
calorimetry_data_2,01.01.2023,cem_b,100,40
```

```python
tam = Measurement(
    folder=datapath,
    metadata_path="metadata.csv",
    metadata_id_column="experiment_nr",
)
# or, for an existing Measurement
tam.add_metadata_source("metadata.csv", "experiment_nr")
```

The metadata is used for plot titles, for
[plotting by category](plotting_example.ipynb#plotting-by-metadata-category)
and for averaging. `get_data_with_metadata()` returns the data with the
metadata columns joined to every row, which is convenient for plotting with
libraries such as seaborn.

## Averaging replicates

Samples that share the same metadata values can be averaged. The heat flow
and heat of every sample are interpolated onto a common time grid
(`bin_width_s`, default 60 s), then mean and standard deviation are computed
per group:

```python
tam.average_by_metadata("cement_name")
tam.plot()  # one curve per cement

tam.undo_average_by_metadata()  # back to the individual samples
```

The averaged data contains the columns `time_s`, `heat_flow_w`, `heat_j`,
`normalized_heat_flow_w_g`, `normalized_heat_j_g`, each with a matching
`*_std` column, and the group name in `sample` and `sample_short`.
Other columns, e.g. the temperature, are dropped. All analysis methods then
work on the averaged curves.

## Downsampling

Files with many data points slow down the analysis. Adaptive downsampling
keeps more points where the curve changes and fewer in flat regions. It is
enabled through the [processing parameters](processing-parameters.md#downsample)
and applied when the `Measurement` is created:

```python
from calocem import ProcessingParameters

processparams = ProcessingParameters()
processparams.downsample.apply = True
processparams.downsample.num_points = 2000

tam = Measurement(folder=datapath, processparams=processparams)
```

## Saving and reloading

Reading and processing many files takes time. A `Measurement` can be saved to
a file and restored in another script:

```python
tam.save("processed_run.pkl")

# later
tam = Measurement.load("processed_run.pkl")
```

The whole object is stored: data, information, metadata and processing
parameters. The file is a convenience cache, not an archive. Files written by
an incompatible CaloCem version cannot be loaded; recreate them from the
original data.

!!! note "Older cache files"
    `Measurement(..., save_cache=True)` writes `_data.pickle` and
    `_info.pickle` to the current working directory, and
    `Measurement(folder, cold_start=False)` reads them back. `save()` and
    `load()` are the recommended replacement, since they use a file name of
    your choice. `remove_pickle_files()` deletes the old cache files.
