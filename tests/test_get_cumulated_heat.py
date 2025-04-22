import pathlib

import pysnooper
import pytest

from calocem import tacalorimetry
from calocem.processparams import ProcessingParameters


@pytest.mark.parametrize(
    "test_input,target_h,expected,expected_deprecated",
    [
        ("calorimetry_data_1.csv", 24, 134.03, 135.71),
        ("calorimetry_data_2.csv", 24, 134.88, 136.59),
    ],
)

@pysnooper.snoop()
def test_get_cumulated_heat(test_input, target_h, expected, expected_deprecated):

    path = pathlib.Path(__file__).parent.parent / "calocem" / "DATA"
    
    processparams = ProcessingParameters()
    processparams.cutoff.cutoff_min = 60
    tam = tacalorimetry.Measurement(path, regex=test_input, auto_clean=False, show_info=True, cold_start=True)

    cumulated_heats = tam.get_cumulated_heat_at_hours(processparams=processparams, target_h=target_h)

    cumulated_heats_deprecated = tam.get_cumulated_heat_at_hours(cutoff_min=30, target_h=target_h)

    assert isinstance(cumulated_heats, tacalorimetry.pd.DataFrame)
    assert round(cumulated_heats.at[0,"cumulated_heat_at_hours"],2) == expected
    assert round(cumulated_heats_deprecated.at[0,"cumulated_heat_at_hours"],2) == expected_deprecated

