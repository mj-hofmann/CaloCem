import matplotlib.pyplot as plt

import sys
from pathlib import Path
from calocem.measurement import Measurement
import calocem.tacalorimetry as tac


datapath = Path(__file__).parent.parent / "calocem" / "DATA"
#datapath = parentfolder / "tmp"

# experiments via class
tam = Measurement(
    folder=datapath,
    # regex = "gen3_calofile.csv",
    regex=r".*no_mass.*",
    # regex= "peak_detection_example1.csv",
    show_info=True,
    auto_clean=False,
    cold_start=True,
)

ax = tam.plot()
ax.set_xlim(0, 25)
ax.set_ylim(0, 5)
plt.show()

