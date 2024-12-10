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
The full documentation is here [here](https://mj-hofmann.github.io/CaloCem/index.html).

## Download Stats

[![PyPI - Downloads](https://img.shields.io/pypi/dm/calocem.svg?color=blue&label=Downloads&logo=pypi&logoColor=gold)](https://pepy.tech/project/calocem)
[![PyPI - Downloads](https://static.pepy.tech/personalized-badge/calocem?period=total&units=none&left_color=black&right_color=grey&left_text=Downloads)](https://pepy.tech/project/tainstcalorimetry)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/calocem.svg?logo=python&label=Python&logoColor=gold)](https://pypi.org/project/calocem/) 

<!-- 
## Example Usage

Import the ```tacalorimetry``` module from **CaloCem**.

```python
# import
import os
from CaloCem import tacalorimetry
```

Next, we define where the exported files are stored. With this information at hand, a ```Measurement``` is initialized. Experimental raw data and the metadata passed in the course of the measurement are retrieved by the methods ```get_data()``` and ```get_information()```, respectively.

```python
# define data path
# "mycalodata" is the subfoldername where the calorimetry
# data files (both .csv or .xlsx) are stored

pathname = os.path.dirname(os.path.realpath(__file__))
path_to_data = pathname + os.sep + "mycalodata"

# Example: if projectfile is at "C:\Users\myname\myproject\myproject.py", then "mydata"
# refers to "C:\Users\myname\myproject\mycalodata" where the data is stored

# load experiments via class, i.e. instantiate tacalorimetry object with data
tam = tacalorimetry.Measurement(folder=path_to_data)

# get sample and information
data = tam.get_data()
info = tam.get_information()
```

### Basic plotting

Furthermore, the ```Measurement``` features a ```plot()```-method for readily visualizing the collected results.

```python
# make plot
tam.plot()
# show plot
tacalorimetry.plt.show()
```

Without further options specified, the ```plot()```-method yields the following.

![enter image description here](https://github.com/mj-hofmann/TAInstCalorimetry/blob/main/tests/plots/Figure%202022-08-08%20112743.png?raw=true)

The ```plot()```-method can also be tuned to show the temporal course of normalized heat. On the one hand, this "tuning" refers to the specification of further keyword arguments such as ```t_unit``` and ```y```. On the other hand, the ```plot()```-method returns an object of type ```matplotlib.axes._subplots.AxesSubplot```, which can be used to further customize the plot. In the following, a guide-to-the-eye line is introduced next to adjuting the axes limts, which is not provided for via the ```plot()```-method's signature.

```python
# show cumulated heat plot
ax = tam.plot(
    t_unit="h",
    y='normalized_heat',
    y_unit_milli=False
)

# define target time
target_h = 1.5

# guide to the eye line
ax.axvline(target_h, color="gray", alpha=0.5, linestyle=":")

# set upper limits
ax.set_ylim(top=250)
ax.set_xlim(right=6)
# show plot
tacalorimetry.plt.show()
```
The following plot is obtained:

![enter image description here](https://github.com/mj-hofmann/TAInstCalorimetry/blob/main/tests/plots/Figure%202022-08-19%20085928.png?raw=true)

### Feature Extraction

Additionally, the package allows among others for streamlining routine tasks such as

- getting cumulated heat values,
- identifying peaks positions and characteristics,
- identifying peak onsets,
- Plotting by Category,
- ... -->

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install TAInstCalorimetry.

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
![Tests](https://github.com/mj-hofmann/TAInstCalorimetry/actions/workflows/run-tests.yml/badge.svg)

## Code Styling

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
