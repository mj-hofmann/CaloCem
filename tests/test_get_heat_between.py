import pathlib

import numpy as np
import pandas as pd
import pytest

from calocem.measurement import Measurement
from calocem.processparams import ProcessingParameters

DATA = pathlib.Path(__file__).parent.parent / "calocem" / "DATA"


@pytest.fixture(scope="module")
def tam():
    return Measurement(
        DATA,
        regex="calorimetry_data_[12].csv",
        auto_clean=False,
        show_info=False,
        cold_start=True,
    )


def sample_data(tam, sample_short):
    data = tam.get_data()
    return data[data["sample_short"] == sample_short].reset_index(drop=True)


def test_heat_between_data_points_is_difference_of_cumulative_heat(tam):
    data = sample_data(tam, "calorimetry_data_1")
    start, end = data.iloc[100], data.iloc[500]

    result = tam.get_heat_between(start["time_s"], end["time_s"], regex="data_1")

    assert len(result) == 1
    assert result.at[0, "heat_j_g"] == pytest.approx(
        end["normalized_heat_j_g"] - start["normalized_heat_j_g"]
    )


def test_heat_between_interpolates_between_data_points(tam):
    data = sample_data(tam, "calorimetry_data_1")
    t0, t1 = data.at[100, "time_s"], data.at[101, "time_s"]
    q0, q1 = data.at[100, "normalized_heat_j_g"], data.at[101, "normalized_heat_j_g"]

    result = tam.get_heat_between(t0, (t0 + t1) / 2, regex="data_1")

    assert result.at[0, "heat_j_g"] == pytest.approx((q1 - q0) / 2)


def test_heat_between_matches_cumulative_heat_at_hours(tam):
    processparams = ProcessingParameters()
    processparams.cutoff.cutoff_min = 60
    cumulative = tam.get_cumulative_heat_at_hours(processparams, target_h=24)

    result = tam.get_heat_between(3600, 24 * 3600)

    np.testing.assert_allclose(
        result["heat_j_g"], cumulative["cumulative_heat_j_g"], atol=0.1
    )


def test_heat_between_per_sample_end_times(tam):
    processparams = ProcessingParameters()
    onsets = tam.get_mainpeak_params(processparams).set_index("sample_short")
    onset = onsets["onset_time_s_from_max_slope"]

    result = tam.get_heat_between(1800, onset).set_index("sample_short")

    assert result["end_s"].to_dict() == pytest.approx(onset.to_dict())
    for sample_short, row in result.iterrows():
        expected = tam.get_heat_between(1800, onset[sample_short], regex=sample_short)
        assert row["heat_j_g"] == pytest.approx(expected.at[0, "heat_j_g"])
        assert row["heat_j_g"] > 0


def test_heat_between_missing_or_invalid_times_give_nan(tam):
    result = tam.get_heat_between(
        {"calorimetry_data_1": 7200}, 3600
    ).set_index("sample_short")

    # end before start
    assert np.isnan(result.at["calorimetry_data_1", "heat_j_g"])
    # no start time for this sample
    assert np.isnan(result.at["calorimetry_data_2", "start_s"])
    assert np.isnan(result.at["calorimetry_data_2", "heat_j_g"])

    outside = tam.get_heat_between(0, 1e9)
    assert outside["heat_j_g"].isna().all()


def test_heat_between_accepts_full_sample_name(tam):
    data = tam.get_data()
    sample = data.loc[data["sample_short"] == "calorimetry_data_1", "sample"].iloc[0]

    by_sample = tam.get_heat_between(1800, {sample: 7200}, regex="data_1")
    by_short = tam.get_heat_between(1800, {"calorimetry_data_1": 7200}, regex="data_1")

    pd.testing.assert_frame_equal(by_sample, by_short)
