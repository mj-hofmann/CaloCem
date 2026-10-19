"""Arguments that have no effect must warn instead of being silently ignored."""

import dataclasses
import pathlib
import warnings

import matplotlib as mpl
import pytest

from calocem.measurement import Measurement
from calocem.processparams import PlottingParameters

DATA_DIR = pathlib.Path(__file__).parent.parent / "calocem" / "DATA"


@pytest.fixture(scope="module")
def tam():
    return Measurement(DATA_DIR, regex=r"calorimetry_data_1\.csv", show_info=False)


@pytest.mark.parametrize(
    "method,kwargs",
    [
        ("get_dormant_period_heatflow", {"cutoff_min": 10}),
        ("get_dormant_period_heatflow", {"prominence": 1e-3}),
        ("get_dormant_period_heatflow", {"show_plot": True}),
        ("get_peak_onsets", {"show_plot": True}),
        ("get_maximum_slope", {"xunit": "h"}),
        ("get_average_slope", {"save_path": pathlib.Path(".")}),
        ("get_peak_onset_via_max_slope", {"intersection": "abscissa"}),
        ("get_peak_onset_via_max_slope", {"time_discarded_s": 600}),
        ("get_astm_c1679_characteristics", {"xscale": "linear"}),
    ],
)
def test_ignored_argument_warns(tam, method, kwargs):
    name = next(iter(kwargs))
    with pytest.warns(DeprecationWarning, match=f"'{name}' parameter has no effect"):
        getattr(tam, method)(**kwargs)


def test_default_call_does_not_warn(tam):
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        tam.get_dormant_period_heatflow()
        tam.get_maximum_slope()


def test_new_code_warns():
    with pytest.warns(DeprecationWarning, match="'new_code'"):
        Measurement(show_info=False, new_code=True)


def test_plot_title_is_dataclass_field():
    names = [f.name for f in dataclasses.fields(PlottingParameters)]
    assert "plot_title" in names
    assert PlottingParameters(plot_title=["cement_name"]).plot_title == ["cement_name"]


def test_import_does_not_change_global_rcparams():
    import calocem.plotting  # noqa: F401

    assert mpl.rcParams["font.family"] == mpl.rcParamsDefault["font.family"]
