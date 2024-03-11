import sys
from pathlib import Path

import matplotlib.pyplot as plt

parentfolder = Path(__file__).parent.parent
sys.path.insert(0, parentfolder.as_posix())

import TAInstCalorimetry.tacalorimetry as ta

datapath = parentfolder / "TAInstCalorimetry" / "DATA"

# experiments via class
tam = ta.Measurement(
    folder=datapath,
    regex=r".*(insitu_bm).*.csv",
    show_info=True,
    auto_clean=False,
    cold_start=True,
)

# apply Tian-correction
tam.apply_tian_correction(
    #tau=[216, 89],
    tau=[235, 75],
    window=13,
    polynom=3,
    spline_smoothing=1e-11,
)

# loop samples
for sample, data in tam.iter_samples():
    p = plt.plot(
        data["time_s"], data["normalized_heat_flow_w_g"], alpha=0.5, linestyle=":"
    )
    plt.plot(
        data["time_s"], data["normalized_heat_flow_w_g_tian"], color=p[0].get_color()
    )

# set limit
plt.xlim(0, 240)
plt.ylim(-.1, 1)
plt.ylabel("normalized_heat_flow")
plt.show()
