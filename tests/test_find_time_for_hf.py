import numpy as np
import pandas as pd
import pytest

from calocem.plotting import SimplePlotter


def test_find_time_for_hf_after_dorm_basic():
    time = np.array([0, 10, 20, 30, 40, 50, 60])
    hf = np.array([0.0, 0.0001, 0.0002, 0.0005, 0.0008, 0.0010, 0.002])
    df = pd.DataFrame({"time_s": time, "normalized_heat_flow_w_g": hf})

    plotter = SimplePlotter()
    # If dorm_time_s is 15, target 0.0005 should be at time 30
    t = plotter._find_time_for_hf_after_dorm(
        df, 0.0005, 15.0, age_col="time_s", target_col="normalized_heat_flow_w_g"
    )
    assert t == pytest.approx(30.0)

    # If dorm_time_s is 35, target 0.0005 should be at time 40 (first value >= target after dorm)
    t2 = plotter._find_time_for_hf_after_dorm(
        df, 0.0005, 35.0, age_col="time_s", target_col="normalized_heat_flow_w_g"
    )
    assert t2 == pytest.approx(40.0)

    # If target not reached after dorm, return None
    t3 = plotter._find_time_for_hf_after_dorm(
        df, 0.003, 0.0, age_col="time_s", target_col="normalized_heat_flow_w_g"
    )
    assert t3 is None
