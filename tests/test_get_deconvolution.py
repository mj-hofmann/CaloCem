import pathlib

import pandas as pd

from calocem.measurement import Measurement
from calocem.processparams import ProcessingParameters


def test_get_deconvolution_default_lognormal_with_configurable_peak_count():
    path = pathlib.Path(__file__).parent.parent / "calocem" / "DATA"

    processparams = ProcessingParameters()
    processparams.cutoff.cutoff_min = 60

    tam = Measurement(
        path,
        regex=r".*calorimetry_data_1\.csv$",
        auto_clean=False,
        show_info=False,
        cold_start=True,
    )

    result = tam.get_deconvolution(processparams=processparams, n_peaks=2, show_plot=False)

    assert isinstance(result, pd.DataFrame)
    assert not result.empty
    assert (result["n_peaks_fitted"] == 2).all()
    assert (result["peak_shape"] == "lognormal").all()
    assert result["component"].nunique() == 2


def test_get_deconvolution_with_chebyshev_baseline():
    path = pathlib.Path(__file__).parent.parent / "calocem" / "DATA"

    processparams = ProcessingParameters()
    processparams.cutoff.cutoff_min = 60
    processparams.deconvolution.chebyshev_degree = 2

    tam = Measurement(
        path,
        regex=r".*calorimetry_data_1\.csv$",
        auto_clean=False,
        show_info=False,
        cold_start=True,
    )

    result = tam.get_deconvolution(
        processparams=processparams,
        n_peaks=2,
        baseline_mode="chebyshev",
        show_plot=False,
    )

    assert isinstance(result, pd.DataFrame)
    assert not result.empty
    assert (result["baseline_mode"] == "chebyshev").all()
    assert result["baseline_cheb_coeffs"].notna().all()


def test_get_left_peak_inflection_tangent_intersection_from_deconvolution():
    path = pathlib.Path(__file__).parent.parent / "calocem" / "DATA"

    processparams = ProcessingParameters()
    processparams.cutoff.cutoff_min = 60

    tam = Measurement(
        path,
        regex=r".*calorimetry_data_1\.csv$",
        auto_clean=False,
        show_info=False,
        cold_start=True,
    )

    deconvolution = tam.get_deconvolution(
        processparams=processparams,
        n_peaks=2,
        peak_shape="lognormal",
        show_plot=False,
    )

    intersections = tam.get_left_peak_inflection_tangent_intersection(
        processparams=processparams,
        deconvolution_results=deconvolution,
        peak_shape="lognormal",
    )

    assert isinstance(intersections, pd.DataFrame)
    assert not intersections.empty
    assert intersections["x_intersection_abscissa_s"].notna().all()
