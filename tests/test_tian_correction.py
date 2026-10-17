import pathlib

import numpy as np

from calocem.measurement import Measurement
from calocem.processparams import ProcessingParameters

DATA_DIR = pathlib.Path(__file__).parent.parent / "calocem" / "DATA"


def _load(processparams=None):
    return Measurement(
        DATA_DIR,
        regex=r"calorimetry_data_1\.csv",
        show_info=False,
        processparams=processparams,
    )


def test_apply_tian_correction_uses_instance_params_by_default():
    tam = _load()
    tam.apply_tian_correction()

    data = tam.get_data()
    for col in [
        "normalized_heat_flow_w_g_tian",
        "gradient_normalized_heat_flow_w_g",
        "normalized_heat_j_g_tian",
    ]:
        assert col in data.columns
    assert np.isfinite(data["normalized_heat_j_g_tian"]).all()


def test_apply_tian_correction_with_tiny_tau_is_identity():
    params = ProcessingParameters()
    params.time_constants.tau1 = 1e-9
    params.time_constants.tau2 = None
    tam = _load(params)
    tam.apply_tian_correction(params)

    data = tam.get_data()
    np.testing.assert_allclose(
        data["normalized_heat_flow_w_g_tian"],
        data["normalized_heat_flow_w_g"],
        rtol=1e-6,
        atol=1e-9,
    )
