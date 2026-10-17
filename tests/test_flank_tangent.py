import pathlib

import numpy as np
import pandas as pd
import pytest

from calocem.measurement import Measurement
from calocem.processparams import ProcessingParameters

DATA_DIR = pathlib.Path(__file__).parent.parent / "calocem" / "DATA"


@pytest.fixture(scope="module")
def tam():
    return Measurement(
        DATA_DIR, regex=r"calorimetry_data_[12]\.csv", show_info=False
    )


def test_get_ascending_flank_tangent_returns_tangent_per_sample(tam):
    result = tam.get_ascending_flank_tangent(ProcessingParameters())

    assert isinstance(result, pd.DataFrame)
    assert sorted(result["sample_short"]) == ["calorimetry_data_1", "calorimetry_data_2"]
    for col in ["tangent_slope", "tangent_time_s", "x_intersection"]:
        assert col in result.columns
    assert (result["tangent_slope"] > 0).all()
    assert np.isfinite(result["x_intersection"]).all()


def test_get_ascending_flank_tangent_kwargs_override_processparams(tam):
    params = ProcessingParameters()
    default = tam.get_ascending_flank_tangent(params)
    overridden = tam.get_ascending_flank_tangent(
        params, flank_fraction_start=0.1, flank_fraction_end=0.3
    )

    assert not np.allclose(default["tangent_slope"], overridden["tangent_slope"])
    # the caller's parameters are not mutated
    assert params.slope_analysis.flank_fraction_start == 0.35


def test_get_ascending_flank_tangent_respects_regex(tam):
    result = tam.get_ascending_flank_tangent(regex="calorimetry_data_1")
    assert list(result["sample_short"]) == ["calorimetry_data_1"]


def test_get_maximum_slope_respects_regex(tam):
    result = tam.get_maximum_slope(regex="calorimetry_data_1")
    assert list(result["sample_short"]) == ["calorimetry_data_1"]
