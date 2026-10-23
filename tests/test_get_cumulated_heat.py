import pathlib

import pandas as pd
#import pysnooper
import pytest

from calocem.measurement import Measurement
from calocem.processparams import ProcessingParameters


@pytest.mark.parametrize(
    "test_input,target_h,expected",
    [
        ("calorimetry_data_1.csv", 24, 134.03),
        ("calorimetry_data_2.csv", 24, 134.88),
    ],
)

#@pysnooper.snoop()
def test_get_cumulated_heat(test_input, target_h, expected):

    path = pathlib.Path(__file__).parent.parent / "calocem" / "DATA"
    
    processparams = ProcessingParameters()
    processparams.cutoff.cutoff_min = 60
    tam = Measurement(path, regex=test_input, auto_clean=False, show_info=True, cold_start=True)

    cumulated_heats = tam.get_cumulative_heat_at_hours(processparams=processparams, target_h=target_h)

    #cumulated_heats_deprecated = tam.get_cumulated_heat_at_hours(cutoff_min=30, target_h=target_h)

    assert isinstance(cumulated_heats, pd.DataFrame)
    assert round(cumulated_heats.at[0,"cumulative_heat_j_g"],2) == expected
    #assert round(cumulated_heats_deprecated.at[0,"cumulated_heat_at_hours"],2) == expected_deprecated



def test_get_cumulated_heat_at_hours_is_deprecated_alias():
    path = pathlib.Path(__file__).parent.parent / "calocem" / "DATA"
    tam = Measurement(path, regex="calorimetry_data_1.csv", auto_clean=False, show_info=False, cold_start=True)

    with pytest.warns(DeprecationWarning, match="get_cumulative_heat_at_hours"):
        old = tam.get_cumulated_heat_at_hours(target_h=24)
    new = tam.get_cumulative_heat_at_hours(target_h=24)

    pd.testing.assert_frame_equal(old, new)
    assert (new["cumulated_heat_at_hours"] == new["cumulative_heat_j_g"]).all()
