import pathlib

import pytest

from calocem.measurement import Measurement


def test_get_maximum_slope():

    # path
    path = pathlib.Path(__file__).parent.parent / "calocem" / "DATA"
    
    # files = "|".join(["calorimetry_data_1.csv", "calorimetry_data_2.csv" ])
    files = [f.stem for f in path.glob("*.csv") if not f.stem.startswith("corrupt")]
    files = "|".join(files)
    # init object
    tam = Measurement(path, auto_clean=False, regex=files, show_info=True)
    

    processparams = tacalorimetry.ProcessingParameters() 
    processparams.spline_interpolation.apply= True
    # get cumulated heats
    max_slopes = tam.get_maximum_slope(processparams)
    # check
    assert isinstance(max_slopes, tacalorimetry.pd.DataFrame)
    assert not max_slopes.empty
    #assert len(max_slopes) == len(tam._data)
