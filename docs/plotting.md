
# Plotting
There are many different use cases for isothermal calorimetry. Here, we focus on the application of isothermal heat flow calorimetry for the hydration of cementitious materials.

## Plot Calorimetry Data

Assume that your calorimetry data is found inside a folder called `calo_data` and your Python script `myscript.py`is in the working directory.
```bash
.
├── myscript.py
└── calo_data
    ├── calofile1.csv
    └── calofile2.csv
```

It is very easy to load the calorimetry files and to plot them. The file `myscript.py` could read like this. First, we create a Path object `datapath` using the pathlib package that is directed at the folder which contains the raw instrument data. Then we pass the the `datapath` object to `ta.Measurement()`. The option `show_info=True` prints the names of the calo files being loaded in the terminal.

```python
from TAInstCalorimetry import tacalorimetry as ta
from pathlib import Path

datapath = Path(__file__).parent / "calo_data"

# create the calorimetry Measurement object
tam = ta.Measurement(
    folder=datapath,
    show_info=True,
    auto_clean=False,
)

# plot the data
tam.plot()

```


## Further Plotting
Blabla