import pathlib

import pandas as pd
#import pysnooper
import pytest

from calocem.measurement import Measurement
from calocem.processparams import ProcessingParameters


@pytest.mark.parametrize(
    "test_input,expected",
    [
        ("calorimetry_data_1.csv", 5.6e-8),
        ("calorimetry_data_2.csv", 6.2e-8),
    ],
)
def test_get_average_slope(test_input, expected):

    path = pathlib.Path(__file__).parent.parent / "calocem" / "DATA"
    
    processparams = ProcessingParameters()
    processparams.cutoff.cutoff_min = 60
    tam = Measurement(path, regex=test_input, auto_clean=False, show_info=True, cold_start=True)

    average_slopes = tam.get_average_slope(processparams=processparams)


    assert isinstance(average_slopes, pd.DataFrame)
    assert round(average_slopes.at[0,"average_slope"],9) == expected

