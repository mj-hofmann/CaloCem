# %%
from pathlib import Path

import matplotlib.pyplot as plt

import CaloCem.tacalorimetry as ta

datapath = Path(__file__).parent.parent / "CaloCem" / "DATA"


# %%
tam2 = ta.Measurement(
    folder=datapath,
    regex=".*bm.*",
    cold_start=True,
    auto_clean=False,
)
# %%

# define processing parameters
processparams = ta.ProcessingParameters()
processparams.tau_values = {"tau1": 300, "tau2": None}

tam2.apply_tian_correction(processparams)

# %%
# loop samples
for sample, data in tam2._iter_samples():
    p = plt.plot(
        data["time_s"],
        data["normalized_heat_flow_w_g"],
        alpha=0.5,
        linestyle=":",
        label="raw",
    )
    # plt.plot(
    #     data["time_s"], data["gradient_normalized_heat_flow_w_g"] * 1e2, label="grad"
    # )
    plt.plot(
        data["time_s"],
        data["normalized_heat_flow_w_g_tian"],
        color=p[0].get_color(),
        label="tian",
    )

plt.xlim(0, 500)
plt.ylabel("normalized_heat_flow")
plt.legend()
plt.show()


# %%

# %%
