import pandas as pd


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
