
#%%
from pathlib import Path

import matplotlib.pyplot as plt

from calocem.tacalorimetry import Measurement
from calocem.processparams import ProcessingParameters

datapath = Path(__file__).parent.parent / "calocem" / "DATA"

# experiments via class
tam = Measurement(
    folder=datapath,
    regex=r".*peak_detection_example[1,3].*",
    show_info=True,
    auto_clean=False,
    cold_start=True,
)
