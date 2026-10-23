import pathlib

import pandas as pd

from calocem.measurement import Measurement
from calocem.processparams import ProcessingParameters

# a few representative files: two CSV layouts, a gen1 file and a sample
# without a main peak (c3a), instead of all files in DATA
SAMPLES = ["c3a", "calorimetry_data_1", "flank_detection1", "gen1_calofile2"]


def test_get_maximum_slope():
    path = pathlib.Path(__file__).parent.parent / "calocem" / "DATA"
    tam = Measurement(path, auto_clean=False, regex="|".join(SAMPLES), show_info=False)

    processparams = ProcessingParameters()
    processparams.spline_interpolation.apply = True
    max_slopes = tam.get_maximum_slope(processparams)

    assert isinstance(max_slopes, pd.DataFrame)
    assert sorted(max_slopes["sample_short"]) == SAMPLES
    assert (max_slopes["gradient_from_max_slope"] > 0).all()

    data = tam.get_data()
    for _, row in max_slopes.iterrows():
        times = data.loc[data["sample_short"] == row["sample_short"], "time_s"]
        assert times.min() <= row["max_slope_time_s"] <= times.max()
