
#%%
from pathlib import Path

from calocem.measurement import Measurement
from calocem.processparams import ProcessingParameters

datapath = Path(__file__).parent.parent / "calocem" / "DATA"

# experiments via class
tam = Measurement(
    folder=datapath,
    regex=r"deconv_example.*",
    show_info=True,
    auto_clean=False,
    cold_start=True,
)

pp = ProcessingParameters()
pp.cutoff.cutoff_min = 60

deconv = tam.get_deconvolution(
    processparams=pp,
    n_peaks=2,
    peak_shape="lognormal",
    show_plot=True,
    baseline_mode="chebyshev",
)

print(deconv[["sample_short", "component", "center_time_s", "amplitude", "fit_r2"]])

left_peak_onset = tam.get_left_peak_inflection_tangent_intersection(
    processparams=pp,
    deconvolution_results=deconv,
)

print(
    left_peak_onset[
        [
            "sample_short",
            "left_peak_center_time_s",
            "inflection_time_s",
            "x_intersection_abscissa_s",
        ]
    ]
)

# %%
