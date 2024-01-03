import pandas as pd
from scipy.interpolate import UnivariateSpline

#
# conversion of DataFrame to float
#
def convert_df_to_float(df: pd.DataFrame) -> pd.DataFrame:
    """
    convert

    Parameters
    ----------
    df : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """

    # type conversion
    for c in df.columns:
        try:
            df[c] = df[c].astype(float)
        except Exception:
            pass

    # return modified DataFrame
    return df


def fit_univariate_spline(df, target_col, s=1e-6):
    """
    fit_univariate_spline

    Parameters
    ----------
    df : pandas dataframe
        DESCRIPTION.

    target_col : str
        DESCRIPTION.

    s : float, optional
        DESCRIPTION. The default is 1e-6.
    
    """
    df[target_col].fillna(0, inplace=True)
    spl = UnivariateSpline(
        df["time_s"], df[target_col], s=s
    )
    df["interpolated"] = spl(df["time_s"])
    return df