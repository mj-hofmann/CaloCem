import pathlib

import pysnooper
import pytest

from calocem import tacalorimetry


#
# run "time check" on each sample file
# see https://www.youtube.com/watch?v=DhUpxWjOhME (~ at 12:20)
# to use the the test:
#       1) install the module locally with "pip install -e ."
#       2) run "pytest" from shell


# @pysnooper.snoop()
def test_get_maximum_slope():

    # path
    path = pathlib.Path(__file__).parent.parent / "calocem" / "DATA"
    
    # files = "|".join(["calorimetry_data_1.csv", "calorimetry_data_2.csv" ])
    files = [f.stem for f in path.glob("*.csv") if not f.stem.startswith("corrupt")]
    files = "|".join(files)
    # init object
    tam = tacalorimetry.Measurement(path, auto_clean=False, regex=files, show_info=True)
    

    processparams = tacalorimetry.ProcessingParameters() 
    processparams.spline_interpolation.apply= True
    # get cumulated heats
    max_slopes = tam.get_maximum_slope(processparams)
    print(max_slopes)
    # check
    assert isinstance(max_slopes, tacalorimetry.pd.DataFrame)
