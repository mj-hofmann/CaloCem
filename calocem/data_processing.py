"""
Data processing operations for calorimetry data.
"""

import logging
import re
from typing import Optional

import numpy as np
import pandas as pd
from scipy import signal
from scipy.ndimage import median_filter

from .exceptions import DataProcessingException
from .processparams import ProcessingParameters

logger = logging.getLogger(__name__)


class DataCleaner:
    """Handles data cleaning operations."""

    @staticmethod
    def auto_clean_data(data: pd.DataFrame) -> pd.DataFrame:
        """
        Remove NaN values and merge differently named columns
        representing the (constant) temperature set for the measurement.
        """
        try:
            cleaned_data = data.copy()

            # Remove NaN values from heat flow columns
            heat_cols = [
                c for c in cleaned_data.columns if re.match("normalized_heat", c)
            ]
            if heat_cols:
                cleaned_data = cleaned_data.dropna(subset=heat_cols).reset_index(
                    drop=True
                )

            # Consolidate temperature columns if they exist
            temp_cols = ["temperature_temperature_c", "temperature_c"]
            if all(col in cleaned_data.columns for col in temp_cols):
                # Calculate NaN count for both temperature columns
                nan_count = cleaned_data["temperature_temperature_c"].isna().astype(
                    int
                ) + cleaned_data["temperature_c"].isna().astype(int)

                # Use values from temperature_c where temperature_temperature_c is NaN
                mask = (cleaned_data["temperature_temperature_c"].isna()) & (
                    nan_count == 1
                )
                cleaned_data.loc[mask, "temperature_temperature_c"] = cleaned_data.loc[
                    (~cleaned_data["temperature_c"].isna()) & (nan_count == 1),
                    "temperature_c",
                ]

                # Remove the redundant temperature_c column
                cleaned_data = cleaned_data.drop(columns=["temperature_c"])

            # Rename consolidated temperature column
            if "temperature_temperature_c" in cleaned_data.columns:
                cleaned_data = cleaned_data.rename(
                    columns={"temperature_temperature_c": "temperature_c"}
                )

            return cleaned_data

        except Exception as e:
            raise DataProcessingException("auto_clean_data", e)

    @staticmethod
    def make_equidistant(data: pd.DataFrame, time_col: str = "time_s") -> pd.DataFrame:
        """Make time series data equidistant by interpolation."""
        try:
            # Create equidistant time array
            time_min = data[time_col].min()
            time_max = data[time_col].max()
            n_points = len(data)

            equidistant_time = np.linspace(time_min, time_max, n_points)

            # Interpolate all numeric columns
            result_data = pd.DataFrame({time_col: equidistant_time})

            for col in data.columns:
                if col != time_col and pd.api.types.is_numeric_dtype(data[col]):
                    # Use linear interpolation
                    result_data[col] = np.interp(
                        equidistant_time, data[time_col], data[col]
                    )
                elif col != time_col:
                    # For non-numeric columns, use forward fill
                    result_data[col] = data[col].iloc[0]

            return result_data

        except Exception as e:
            raise DataProcessingException("make_equidistant", e)


class HeatFlowProcessor:
    """Processes heat flow data according to processing parameters."""

    def __init__(self, processparams: ProcessingParameters):
        self.processparams = processparams

    def apply_rolling_mean(
        self, data: pd.DataFrame, target_col: str = "normalized_heat_flow_w_g"
    ) -> pd.DataFrame:
        """Apply rolling mean smoothing to data."""
        try:
            result_data = data.copy()
            if (
                self.processparams.rolling_mean.apply
                and self.processparams.rolling_mean.window > 1
            ):
                result_data[target_col] = (
                    result_data[target_col]
                    .rolling(window=self.processparams.rolling_mean.window, center=True)
                    .mean()
                )
                # Fill NaN values at edges
                result_data[target_col] = result_data[target_col].bfill().ffill()

            return result_data

        except Exception as e:
            raise DataProcessingException("apply_rolling_mean", e)

    def calculate_heatflow_derivatives(
        self,
        data: pd.DataFrame,
        time_col: str = "time_s",
        target_col: str = "normalized_heat_flow_w_g",
    ) -> tuple[np.ndarray, np.ndarray]:
        """Calculate gradient and curvature of heat flow data."""
        try:
            time = data[time_col].to_numpy()
            heat_flow = data[target_col].to_numpy()

            # Calculate first derivative (gradient)
            gradient = np.gradient(heat_flow, time)

            # Calculate second derivative (curvature)
            curvature = np.gradient(gradient, time)

            return gradient, curvature

        except Exception as e:
            raise DataProcessingException("calculate_heatflow_derivatives", e)

    def get_largest_slope(
        self, data: pd.DataFrame, processparams: ProcessingParameters
    ) -> pd.DataFrame:
        """Find the point with the largest slope in the data."""
        try:
            # Apply filtering if specified
            filtered_data = data.copy()

            # Apply cutoff time
            if processparams.cutoff.cutoff_min:
                cutoff_seconds = processparams.cutoff.cutoff_min * 60
                filtered_data = filtered_data[filtered_data["time_s"] >= cutoff_seconds]

            if filtered_data.empty:
                return pd.DataFrame()

            # Find maximum gradient
            max_gradient_idx = filtered_data["gradient"].idxmax()

            if pd.isna(max_gradient_idx):
                return pd.DataFrame()

            # Return the row with maximum gradient
            result = filtered_data.loc[[max_gradient_idx]].copy()

            return result

        except Exception as e:
            raise DataProcessingException("get_largest_slope", e)

    def apply_median_filter(
        self, data: pd.DataFrame, target_col: str = "normalized_heat_flow_w_g"
    ) -> pd.DataFrame:
        """Apply median filter to reduce noise."""
        try:
            result_data = data.copy()
            if self.processparams.median_filter.apply:
                result_data[target_col] = median_filter(
                    result_data[target_col], size=3  # Default kernel size
                )

            return result_data

        except Exception as e:
            raise DataProcessingException("apply_median_filter", e)


class PeakDetector:
    """Detects peaks in calorimetry data."""

    def __init__(self, processparams: ProcessingParameters):
        self.processparams = processparams

    def find_peaks(
        self, data: pd.DataFrame, target_col: str = "normalized_heat_flow_w_g"
    ) -> tuple[np.ndarray, dict]:
        """Find peaks in the heat flow data."""
        try:
            # Apply cutoff if specified
            filtered_data = data.copy()
            if self.processparams.cutoff.cutoff_min:
                cutoff_seconds = self.processparams.cutoff.cutoff_min * 60
                filtered_data = filtered_data[filtered_data["time_s"] >= cutoff_seconds]

            if filtered_data.empty:
                return np.array([]), {}

            # Reset index to ensure peak indices align with filtered data
            filtered_data = filtered_data.reset_index(drop=True)

            # Find peaks using scipy
            peaks, properties = signal.find_peaks(
                filtered_data[target_col],
                prominence=self.processparams.peakdetection.prominence,
                distance=self.processparams.peakdetection.distance,
            )

            return peaks, properties

        except Exception as e:
            raise DataProcessingException("find_peaks", e)

    def compile_peak_characteristics(
        self, data: pd.DataFrame, peaks: np.ndarray, properties: dict
    ) -> pd.DataFrame:
        """Compile peak characteristics into a DataFrame."""
        try:
            if len(peaks) == 0:
                return pd.DataFrame()

            # Get peak data
            peak_data = data.iloc[peaks, :].copy()

            # Add prominence information
            prominence_df = pd.DataFrame(
                properties["prominences"], index=peaks, columns=["prominence"]
            )

            # Add peak numbers
            peak_numbers = pd.DataFrame({"peak_nr": np.arange(len(peaks))}, index=peaks)

            # Combine all characteristics
            peak_characteristics = pd.concat(
                [peak_data, prominence_df, peak_numbers], axis=1
            )

            return peak_characteristics

        except Exception as e:
            raise DataProcessingException("compile_peak_characteristics", e)


class OnsetDetector:
    """Detects reaction onsets in calorimetry data."""

    def find_gradient_onset(
        self,
        data: pd.DataFrame,
        time_col: str = "time_s",
        target_col: str = "normalized_heat_flow_w_g",
        time_discarded_s: float = 900,
        rolling: int = 1,
        gradient_threshold: float = 0.0005,
    ) -> pd.DataFrame:
        """Find onset based on gradient threshold."""
        try:
            # Calculate gradient
            smoothed_data = data[target_col].rolling(rolling).mean()
            gradient = np.gradient(smoothed_data, data[time_col])

            # Add gradient to data
            result_data = data.copy()
            result_data["gradient"] = gradient

            # Apply time cutoff
            characteristics = result_data[result_data[time_col] >= time_discarded_s]

            # Find first point above gradient threshold
            onset_candidates = characteristics[
                characteristics["gradient"] > gradient_threshold
            ]

            if onset_candidates.empty:
                return pd.DataFrame()

            # Return first onset
            onset = onset_candidates.head(1)

            return onset

        except Exception as e:
            raise DataProcessingException("find_gradient_onset", e)


class IntersectionCalculator:
    """Calculates intersections for onset detection."""

    @staticmethod
    def calculate_dormant_hf_intersection(
        max_slope_row: pd.Series, dormant_hf: float
    ) -> float:
        """Calculate intersection with dormant heat flow line."""
        try:
            # Calculate y-intercept of tangent line: y = mx + b
            # b = y1 - m*x1
            y_intercept = (
                max_slope_row["normalized_heat_flow_w_g"]
                - max_slope_row["time_s"] * max_slope_row["gradient"]
            )

            # Find intersection: dormant_hf = m*x + b
            # x = (dormant_hf - b) / m
            x_intersect = (dormant_hf - y_intercept) / max_slope_row["gradient"]

            return x_intersect

        except Exception as e:
            raise DataProcessingException("calculate_dormant_hf_intersection", e)

    @staticmethod
    def calculate_abscissa_intersection(max_slope_row: pd.Series) -> float:
        """Calculate intersection with x-axis (y=0)."""
        try:
            # For line y = mx + b, when y=0: x = -b/m
            # Where b = y1 - m*x1
            x_intersect = max_slope_row["time_s"] - (
                max_slope_row["normalized_heat_flow_w_g"] / max_slope_row["gradient"]
            )

            return x_intersect

        except Exception as e:
            raise DataProcessingException("calculate_abscissa_intersection", e)


class SampleIterator:
    """Utility for iterating over samples in data."""

    @staticmethod
    def iter_samples(data: pd.DataFrame, regex: Optional[str] = None):
        """
        Iterate samples and return corresponding data.

        Yields:
            sample (str): Name of the current sample
            sample_data (pd.DataFrame): Data corresponding to the current sample
        """
        
        #if pd.isna(data["sample"]).all():
        #    data["sample"] = data["sample_short"]
        data["sample"] = data["sample"].fillna(data["sample_short"])

        for sample, sample_data in data.groupby(by="sample"):
            if regex:
                if not re.search(regex, str(sample)):
                    continue

            yield sample, sample_data


class DataNormalizer:
    """Handles data normalization operations."""

    @staticmethod
    def normalize_sample_to_mass(
        data: pd.DataFrame, sample_short: str, mass_g: float, show_info: bool = True
    ) -> pd.DataFrame:
        """Normalize heat flow values to a specific mass."""
        try:
            result_data = data.copy()

            # Find rows for the specified sample
            sample_mask = result_data["sample_short"] == sample_short

            if not sample_mask.any():
                if show_info:
                    print(f"Sample '{sample_short}' not found in data")
                return result_data

            # Normalize heat flow columns
            heat_flow_cols = [col for col in result_data.columns if "heat_flow" in col]
            heat_cols = [col for col in result_data.columns if col.startswith("heat_j")]

            for col in heat_flow_cols:
                if "normalized" not in col:
                    normalized_col = f"normalized_{col}_g"
                    result_data.loc[sample_mask, normalized_col] = (
                        result_data.loc[sample_mask, col] / mass_g
                    )

            for col in heat_cols:
                if "normalized" not in col:
                    normalized_col = f"normalized_{col}_g"
                    result_data.loc[sample_mask, normalized_col] = (
                        result_data.loc[sample_mask, col] / mass_g
                    )

            if show_info:
                print(f"Sample '{sample_short}' normalized to {mass_g}g")

            return result_data

        except Exception as e:
            raise DataProcessingException("normalize_sample_to_mass", e)

    @staticmethod
    def infer_heat_j_column(data: pd.DataFrame) -> pd.DataFrame:
        """Infer missing heat_j columns from available data."""
        try:
            result_data = data.copy()

            # Check if heat_j column needs to be inferred
            if (
                "heat_j" not in result_data.columns
                and "heat_flow_w" in result_data.columns
            ):
                # Use cumulative integration of heat flow
                from scipy import integrate

                for sample, sample_data in result_data.groupby("sample"):
                    sample_mask = result_data["sample"] == sample
                    heat_j = integrate.cumulative_trapezoid(
                        sample_data["heat_flow_w"], x=sample_data["time_s"], initial=0
                    )
                    result_data.loc[sample_mask, "heat_j"] = heat_j

            return result_data

        except Exception as e:
            raise DataProcessingException("infer_heat_j_column", e)
