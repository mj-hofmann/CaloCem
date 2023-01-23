# TAInstCalorimetry- Interfacing with experimental results file from TAM Air calorimeters made easy.

After collecting multiple experimental results files from a TAM Air calorimeter you will be left with multiple *.xls*-files obtained as exports from the device control software. To achieve a side by side comparison of theses results and some basic extraction of relevant parameters, **TAInstCalorimetry** is here to get this done smoothly.

*Note: **TAInstCalorimetry** has been developed without involvement of **TA Instruments** and is thus independent from the company and its software.*

## Info / Downloads

[![PyPI - Downloads](https://img.shields.io/pypi/dm/tainstcalorimetry.svg?color=blue&label=Downloads&logo=pypi&logoColor=gold)](https://pepy.tech/project/tainstcalorimetry)
[![PyPI - Downloads](https://static.pepy.tech/personalized-badge/tainstcalorimetry?period=total&units=none&left_color=black&right_color=grey&left_text=Downloads)](https://pepy.tech/project/tainstcalorimetry)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/tainstcalorimetry.svg?logo=python&label=Python&logoColor=gold)](https://pypi.org/project/tainstcalorimetry/) 

## Example Usage

Import the ```tacalorimetry``` module from **TAInstCalorimetry**.

```python
# import
import os
from TAInstCalorimetry import tacalorimetry
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

Furthermore, the ```Measurement``` features a ```plot()```-method for readily visualizing the collected results.

```python
# make plot
tam.plot()
# show plot
tacalorimetry.plt.show()
```

Without further options specified, the ```plot()```-method yields the following.

![enter image description here](https://github.com/mj-hofmann/TAInstCalorimetry/blob/main/tests/plots/Figure%202022-08-08%20112743.png?raw=true)

The cumulated heat after a certain period of time from starting the measurement is a relevant quantity for answering different types of questions. For this purpose, the method ```get_cumulated_heat_at_hours``` returns on overview of this parameter for all the samples in the specified folder.

```python
# get table of cumulated heat at certain age
cum_h = tam.get_cumulated_heat_at_hours(
                target_h=2, 
                cutoff_min=0
                )
# show result
print(cum_h)
```
The ```plot()```-method can also be tuned to show the temporal course of normalized heat.

```python
# show cumulated heat plot
ax = tam.plot(
    t_unit="h",
    y='normalized_heat',
    y_unit_milli=False
)

# guide to the eye line
ax.axvline(2, color="gray", alpha=0.5, linestyle=":")

# set upper limits
ax.set_ylim(top=250)
ax.set_xlim(right=6)
# show plot
tacalorimetry.plt.show())
```
The following plot is obtained:

![enter image description here](https://github.com/mj-hofmann/TAInstCalorimetry/blob/main/tests/plots/Figure%202022-08-19%20085928.png?raw=true)

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install TAInstCalorimetry.

```bash
pip install TAInstCalorimetry
```

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

List of contributors:
- tgaedt
- mj-hofmann

## License
[GNU GPLv3](https://choosealicense.com/licenses/gpl-3.0/#)


## Test
![Tests](https://github.com/mj-hofmann/TAInstCalorimetry/actions/workflows/run-tests.yml/badge.svg)

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)