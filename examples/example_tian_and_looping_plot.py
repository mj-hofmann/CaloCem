# %%
from pathlib import Path

import matplotlib.pyplot as plt

from calocem import Measurement, ProcessingParameters


datapath = Path(__file__).parent.parent / "calocem" / "DATA"


# %%
tam = Measurement(
    folder=datapath,
    regex=".*bm.*",
    cold_start=True,
    auto_clean=False,
)

# %%
# Single time constant: correct for thermal lag using only tau1
# Set tau2 = None to explicitly use first-order Tian correction:
#   hf_corrected = dHF/dt * tau1 + HF
processparams = ProcessingParameters()
processparams.time_constants.tau1 = 300
processparams.time_constants.tau2 = None

tam.apply_tian_correction(processparams)

# %%
# Loop over samples and overlay raw vs Tian-corrected heat flow
fig, ax = plt.subplots()

for sample, data in tam.get_data().groupby("sample_short"):
    (line,) = ax.plot(
        data["time_s"] / 3600,
        data["normalized_heat_flow_w_g"] * 1e3,
        alpha=0.5,
        linestyle=":",
        label=f"{sample} (raw)",
    )
    ax.plot(
        data["time_s"] / 3600,
        data["normalized_heat_flow_w_g_tian"] * 1e3,
        color=line.get_color(),
        label=f"{sample} (Tian, tau1={processparams.time_constants.tau1}s)",
    )

ax.set_xlabel("Time / h")
ax.set_ylabel("Normalized heat flow / mW g$^{-1}$")
ax.set_xlim(0, 1)
ax.legend()
ax.set_title("Tian correction — single time constant")
plt.tight_layout()
plt.show()

# %%
# Dual time constants: second-order Tian correction:
#   hf_corrected = dHF/dt * (tau1 + tau2) + d²HF/dt² * tau1*tau2 + HF
processparams2 = ProcessingParameters()
processparams2.time_constants.tau1 = 300
processparams2.time_constants.tau2 = 100

tam.apply_tian_correction(processparams2)

# %%
fig, ax = plt.subplots()

for sample, data in tam.get_data().groupby("sample_short"):
    (line,) = ax.plot(
        data["time_s"] / 3600,
        data["normalized_heat_flow_w_g"] * 1e3,
        alpha=0.5,
        linestyle=":",
        label=f"{sample} (raw)",
    )
    ax.plot(
        data["time_s"] / 3600,
        data["normalized_heat_flow_w_g_tian"] * 1e3,
        color=line.get_color(),
        label=f"{sample} (Tian, tau1={processparams2.time_constants.tau1}s, tau2={processparams2.time_constants.tau2}s)",
    )

ax.set_xlabel("Time / h")
ax.set_ylabel("Normalized heat flow / mW g$^{-1}$")
ax.set_xlim(0, 1)
ax.legend()
ax.set_title("Tian correction — dual time constants")
plt.tight_layout()
plt.show()
