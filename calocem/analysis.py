"""
Analysis operations for calorimetry data.
"""

import logging
import pathlib
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from .data_processing import (
    HeatFlowProcessor,
    IntersectionCalculator,
    OnsetDetector,
    PeakDetector,
    SampleIterator,
)
from .exceptions import DataProcessingException
from .processparams import ProcessingParameters

logger = logging.getLogger(__name__)


@dataclass
class OnsetCharacteristics:
    """Data class to hold onset calculation results."""

    sample: str
    time_s: float
    normalized_heat_flow_w_g: float
    gradient: float
    dorm_time_s: float
    normalized_heat_flow_w_g_dormant: float
    x_intersect: float
    intersection_type: str
    xunit: str


class PeakAnalyzer:
    """Analyzes peaks in calorimetry data."""

    def __init__(self, processparams: ProcessingParameters):
        self.processparams = processparams
        self.peak_detector = PeakDetector(processparams)

    def get_peaks(
        self,
        data: pd.DataFrame,
        target_col: str = "normalized_heat_flow_w_g",
        regex: Optional[str] = None,
    ) -> pd.DataFrame:
        """Get DataFrame of peak characteristics."""
        try:
            list_of_peaks_dfs = []

            for sample, sample_data in SampleIterator.iter_samples(data, regex):
                # Apply cutoff
                filtered_data = sample_data.copy()
                if self.processparams.cutoff.cutoff_min:
                    cutoff_seconds = self.processparams.cutoff.cutoff_min * 60
                    filtered_data = filtered_data[
                        filtered_data["time_s"] >= cutoff_seconds
                    ]

                filtered_data = filtered_data.reset_index(drop=True)

                # Find peaks
                peaks, properties = self.peak_detector.find_peaks(
                    filtered_data, target_col
                )

                if len(peaks) > 0:
                    # Compile peak characteristics
                    peak_characteristics = (
                        self.peak_detector.compile_peak_characteristics(
                            filtered_data, peaks, properties
                        )
                    )
                    list_of_peaks_dfs.append(peak_characteristics)

            if list_of_peaks_dfs:
                return pd.concat(list_of_peaks_dfs, ignore_index=True)
            else:
                return pd.DataFrame()

        except Exception as e:
            raise DataProcessingException("get_peaks", e)


class SlopeAnalyzer:
    """Analyzes maximum slopes in calorimetry data."""

    def __init__(self, processparams: ProcessingParameters):
        self.processparams = processparams
        self.processor = HeatFlowProcessor(processparams)

    def get_maximum_slope(
        self,
        data: pd.DataFrame,
        target_col: str = "normalized_heat_flow_w_g",
        age_col: str = "time_s",
        time_discarded_s: float = 900,
        exclude_discarded_time: bool = False,
        regex: Optional[str] = None,
        #read_start_c3s: bool = False,
        #metadata: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """Find the point in time of the maximum slope and calculate gradient."""
        try:
            list_of_characteristics = []
            time_discarded_s = self.processparams.cutoff.cutoff_min * 60 if self.processparams.cutoff.cutoff_min else 0
            for sample, sample_data in SampleIterator.iter_samples(data, regex):
                sample_name = pathlib.Path(str(sample)).stem

                # Apply time filtering
                filtered_data = sample_data.copy()
                if time_discarded_s > 0:
                    filtered_data = filtered_data[
                        filtered_data[age_col] >= time_discarded_s
                    ]

                # Manual C3S time definition if requested
                # if read_start_c3s and metadata is not None:
                #     try:
                #         c3s_start = metadata.query(f"sample_number == '{sample_name}'")[
                #             "t_c3s_min_s"
                #         ].values[0]
                #         c3s_end = metadata.query(f"sample_number == '{sample_name}'")[
                #             "t_c3s_max_s"
                #         ].values[0]
                #         filtered_data = filtered_data[
                #             (filtered_data[age_col] >= c3s_start)
                #             & (filtered_data[age_col] <= c3s_end)
                #         ]
                #     except (IndexError, KeyError):
                #         logger.warning(f"No C3S time data found for {sample_name}")

                if filtered_data.empty:
                    continue

                # Make data equidistant and process
                from .data_processing import DataCleaner

                filtered_data = DataCleaner.make_equidistant(filtered_data)

                # Apply rolling mean if specified
                if self.processparams.rolling_mean.apply:
                    filtered_data = self.processor.apply_rolling_mean(
                        filtered_data, target_col
                    )

                # Calculate derivatives
                gradient, curvature = self.processor.calculate_heatflow_derivatives(
                    filtered_data, age_col, target_col
                )
                filtered_data["gradient"] = gradient
                filtered_data["curvature"] = curvature

                # Find maximum slope
                characteristics = self.processor.get_largest_slope(
                    filtered_data, self.processparams
                )

                if not characteristics.empty:
                    list_of_characteristics.append(characteristics)

            if list_of_characteristics:
                return pd.concat(list_of_characteristics, ignore_index=True)
            else:
                logger.warning("No maximum slope found, check processing parameters")
                return pd.DataFrame()

        except Exception as e:
            raise DataProcessingException("get_maximum_slope", e)


class OnsetAnalyzer:
    """Analyzes peak onsets in calorimetry data."""

    def __init__(self, processparams: ProcessingParameters):
        self.processparams = processparams
        self.onset_detector = OnsetDetector()
        self.intersection_calc = IntersectionCalculator()

    def get_peak_onsets(
        self,
        data: pd.DataFrame,
        target_col: str = "normalized_heat_flow_w_g",
        age_col: str = "time_s",
        time_discarded_s: float = 900,
        rolling: int = 1,
        gradient_threshold: float = 0.0005,
        exclude_discarded_time: bool = False,
        regex: Optional[str] = None,
    ) -> pd.DataFrame:
        """Get peak onsets based on gradient threshold criterion."""
        try:
            list_of_characteristics = []

            for sample, sample_data in SampleIterator.iter_samples(data, regex):
                # Apply time filtering
                if exclude_discarded_time:
                    sample_data = sample_data[sample_data[age_col] >= time_discarded_s]

                sample_data = sample_data.reset_index(drop=True)

                # Find onset
                characteristics = self.onset_detector.find_gradient_onset(
                    sample_data,
                    age_col,
                    target_col,
                    time_discarded_s,
                    rolling,
                    gradient_threshold,
                )

                if not characteristics.empty:
                    list_of_characteristics.append(characteristics)

            if list_of_characteristics:
                return pd.concat(list_of_characteristics, ignore_index=True)
            else:
                return pd.DataFrame()

        except Exception as e:
            raise DataProcessingException("get_peak_onsets", e)

    def get_peak_onset_via_max_slope(
        self,
        data: pd.DataFrame,
        max_slopes: pd.DataFrame,
        dormant_hfs: pd.DataFrame,
        #intersection: str = "dormant_hf",
        #xunit: str = "s",
    ) -> pd.DataFrame:
        """Calculate peak onset via maximum slope intersection method."""
        try:
            list_characteristics = []

            for _, row in max_slopes.iterrows():
                sample_short = row["sample_short"]

                # Find corresponding dormant heat flow
                sample_dorm_hf = dormant_hfs[
                    dormant_hfs["sample_short"] == sample_short
                ]
                if sample_dorm_hf.empty:
                    continue

                dorm_hf_value = float(
                    sample_dorm_hf["normalized_heat_flow_w_g_dormant"].iloc[0]
                )

                # Calculate intersection with tangent to dormant heat flow
                #if intersection == "dormant_hf":
                x_intersect = (
                    self.intersection_calc.calculate_dormant_hf_intersection(
                            row, dorm_hf_value
                    )
                )
                # intersection with abscissa
                x_intersect_abscissa = (
                        self.intersection_calc.calculate_abscissa_intersection(row)
                    )

                # Get sample data for limits (not used in this simplified version)
                sample_data = data[data["sample_short"] == sample_short]
                if sample_data.empty:
                    continue

                # Append characteristics
                list_characteristics.append(
                    {
                        "sample": sample_short,
                        "onset_time_s": x_intersect,
                        "onset_time_min": x_intersect / 60,
                        "onset_time_s_abscissa": x_intersect_abscissa,
                    }
                )

            if list_characteristics:
                onsets = pd.DataFrame(list_characteristics)

                # Merge with dormant heat flow data
                merge_cols = [
                    "sample_short",
                    "normalized_heat_flow_w_g",
                    "normalized_heat_j_g",
                ]
                available_cols = [
                    col for col in merge_cols if col in dormant_hfs.columns
                ]

                if available_cols:
                    onsets = onsets.merge(
                        dormant_hfs[available_cols],
                        left_on="sample",
                        right_on="sample_short",
                        how="left",
                    )

                    # Rename columns for clarity
                    rename_map = {
                        "normalized_heat_flow_w_g": "normalized_heat_flow_w_g_at_dorm_min",
                        "normalized_heat_j_g": "normalized_heat_j_g_at_dorm_min",
                    }
                    onsets = onsets.rename(columns=rename_map)

                return onsets
            else:
                return pd.DataFrame()

        except Exception as e:
            raise DataProcessingException("get_peak_onset_via_max_slope", e)


class DormantPeriodAnalyzer:
    """Analyzes dormant period heat flow."""

    def __init__(self, processparams: ProcessingParameters):
        self.processparams = processparams

    def get_dormant_period_heatflow(
        self,
        data: pd.DataFrame,
        peaks: pd.DataFrame,
        regex: Optional[str] = None,
        upper_dormant_thresh_w_g: float = 0.002,
    ) -> pd.DataFrame:
        """Get dormant period heat flow characteristics."""
        try:
            list_dfs = []

            for sample, sample_data in SampleIterator.iter_samples(data, regex):
                sample_short = pathlib.Path(str(sample)).stem

                if peaks.empty:
                    continue

                # Get corresponding peaks
                sample_peaks = peaks[peaks["sample_short"] == sample_short]

                # Apply cutoff time
                filtered_data = sample_data.copy()
                if self.processparams.cutoff.cutoff_min:
                    cutoff_seconds = self.processparams.cutoff.cutoff_min * 60
                    filtered_data = filtered_data[
                        filtered_data["time_s"] >= cutoff_seconds
                    ]

                # Limit data to before first peak if peaks exist
                if not sample_peaks.empty:
                    first_peak_time = sample_peaks["time_s"].min()
                    filtered_data = filtered_data[
                        filtered_data["time_s"] <= first_peak_time
                    ]

                if filtered_data.empty:
                    continue

                # Reset index and find minimum heat flow
                filtered_data = filtered_data.reset_index(drop=True)
                min_idx = int(filtered_data["normalized_heat_flow_w_g"].idxmin())
                dormant_data = filtered_data.iloc[min_idx : min_idx + 1]

                list_dfs.append(dormant_data)

            if list_dfs:
                dormant_hf = pd.concat(list_dfs, ignore_index=True)
                dormant_hf = self.rename_and_select_columns(dormant_hf)
                return dormant_hf
            #pd.concat([dormant_hf, self.rename_and_select_columns(dormant_hf)], ignore_index=True)
            else:
                return pd.DataFrame()

        except Exception as e:
            raise DataProcessingException("get_dormant_period_heatflow", e)

    def rename_and_select_columns(
        self, dormant_hf: pd.DataFrame
    ) -> pd.DataFrame:
        """Rename and select relevant columns for dormant heat flow DataFrame."""
        if dormant_hf.empty:
            return dormant_hf

        rename_map = {
            "time_s": "dorm_time_s",
            "normalized_heat_flow_w_g": "normalized_heat_flow_w_g_dormant",
            "normalized_heat_j_g": "normalized_heat_j_g_dormant",
        }
        cols_to_select = ["sample", "sample_short"] + list(rename_map.keys())

        available_cols = [col for col in cols_to_select if col in dormant_hf.columns]
        reduced_df = dormant_hf[available_cols].rename(columns=rename_map)
        return reduced_df

class ASTMC1679Analyzer:
    """Analyzes characteristics according to ASTM C1679."""

    def __init__(self, processparams: ProcessingParameters):
        self.processparams = processparams

    def get_astm_c1679_characteristics(
        self,
        data: pd.DataFrame,
        peaks: pd.DataFrame,
        individual: bool = True,
        regex: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Get characteristics according to ASTM C1679.

        Compiles a list of data points at half-maximum normalized heat flow,
        where half maximum is determined individually or as mean value.
        """
        try:
            # Sort peaks by ascending normalized heat flow and select highest peak per sample
            sorted_peaks = peaks.sort_values(
                by="normalized_heat_flow_w_g", ascending=True
            )
            highest_peaks = sorted_peaks.groupby(by="sample").last()

            astm_times = []

            for sample, sample_data in SampleIterator.iter_samples(data, regex):
                sample_short = pathlib.Path(str(sample)).stem

                # Check if peak was found for this sample
                sample_peak = highest_peaks[
                    highest_peaks["sample_short"] == sample_short
                ]

                if sample_peak.empty:
                    logger.warning(f"No peak found for sample {sample_short}")
                    continue

                # Get peak height
                peak_height = float(sample_peak["normalized_heat_flow_w_g"].iloc[0])

                # Calculate target half-maximum value
                if individual:
                    target_value = peak_height / 2
                else:
                    # Use mean of all peak heights
                    mean_peak_height = highest_peaks["normalized_heat_flow_w_g"].mean()
                    target_value = mean_peak_height / 2

                # get time_s for half the maximum heat flow, i.e. find the time for which normalized_heat_flow_w_g <= target_value
                #peak_time_s = highest_peaks.query(f"sample == {sample}")#["time_s"].values[0]
                peak_time_s = float(sample_peak["time_s"].iloc[0])
                sample_data_filtered = sample_data.query(f"time_s < {peak_time_s} and normalized_heat_flow_w_g <= {target_value}")
                # Get heat at target time   

                # sample_data_filtered = sample_data[
                #     sample_data["normalized_heat_flow_w_g"] <= target_value
                # ]

                if not sample_data_filtered.empty:
                    astm_time = sample_data_filtered.tail(1)
                    astm_times.append(astm_time)
                else:
                    logger.warning(
                        f"No half-maximum point found for sample {sample_short}"
                    )

            if astm_times:
                df = pd.concat(astm_times, ignore_index=True) 
                df = self.rename_and_select_columns(df)
                return df
            else:
                return pd.DataFrame()

        except Exception as e:
            raise DataProcessingException("get_astm_c1679_characteristics", e)

    def rename_and_select_columns(
        self, astm_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Rename and select relevant columns for ASTM DataFrame."""
        if astm_df.empty:
            return astm_df

        rename_map = {
            "time_s": "astm_time_s",
            "normalized_heat_flow_w_g": "normalized_heat_flow_w_g_astm",
            "normalized_heat_j_g": "normalized_heat_j_g_astm",
        }
        cols_to_select = ["sample", "sample_short"] + list(rename_map.keys())

        available_cols = [col for col in cols_to_select if col in astm_df.columns]
        reduced_df = astm_df[available_cols].rename(columns=rename_map)
        return reduced_df

class HeatCalculator:
    """Calculates cumulative heat at specific times."""

    @staticmethod
    def get_cumulated_heat_at_hours(
        data: pd.DataFrame, target_h: float = 4, cutoff_min: Optional[float] = None
    ) -> pd.DataFrame:
        """Get cumulated heat flow at a specific age."""
        try:

            def calculate_heat_at_target(sample_data: pd.DataFrame) -> float:
                # Convert target time to seconds
                target_s = 3600 * target_h

                # Get heat at target time
                target_data = sample_data[sample_data["time_s"] <= target_s]
                if target_data.empty:
                    return np.nan

                heat_at_target = float(target_data["normalized_heat_j_g"].iloc[-1])

                # Apply cutoff if specified
                if cutoff_min:
                    cutoff_s = 60 * cutoff_min
                    try:
                        cutoff_data = sample_data[sample_data["time_s"] <= cutoff_s]
                        if not cutoff_data.empty:
                            heat_at_cutoff = float(
                                cutoff_data["normalized_heat_j_g"].iloc[-1]
                            )
                            heat_at_target -= heat_at_cutoff
                    except (IndexError, ValueError):
                        pass

                return heat_at_target

            # Group by sample and apply calculation
            results = (
                data.groupby(by="sample")[["time_s", "normalized_heat_j_g"]]
                .apply(calculate_heat_at_target)
                .reset_index()
            )

            results.columns = ["sample", "cumulated_heat_at_hours"]
            results["target_h"] = target_h
            results["cutoff_min"] = cutoff_min

            return results

        except Exception as e:
            raise DataProcessingException("get_cumulated_heat_at_hours", e)


class AverageSlopeAnalyzer:
    """Analyzes average slopes between onset and heat flow maximum."""

    def __init__(self, processparams: ProcessingParameters):
        self.processparams = processparams

    def get_average_slope(
        self,
        data: pd.DataFrame,
        max_slopes: pd.DataFrame,
        onsets: pd.DataFrame,
        target_col: str = "normalized_heat_flow_w_g",
        age_col: str = "time_s",
        regex: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Calculate average slope by determining 4 additional slope values
        between onset time and heat flow maximum.
        """
        try:
            list_of_characteristics = []

            for sample, sample_data in SampleIterator.iter_samples(data, regex):
                sample_short = pathlib.Path(str(sample)).stem

                # Get max slope data for this sample
                max_slope_row = max_slopes[max_slopes["sample_short"] == sample_short]
                if max_slope_row.empty:
                    continue

                # Get onset data for this sample
                onset_row = onsets[onsets["sample_short"] == sample_short]
                if onset_row.empty:
                    continue

                # Get time points
                onset_time = onset_row["onset_time_s_from_max_slope"].iloc[0]
                max_slope_time = max_slope_row[age_col].iloc[0]

                # Find heat flow maximum after onset
                data_after_onset = sample_data[sample_data[age_col] >= onset_time]
                if data_after_onset.empty:
                    continue

                max_hf_time = data_after_onset.loc[
                    data_after_onset[target_col].idxmax(), age_col
                ]

                # Create 4 intermediate time points between onset and heat flow maximum
                if max_hf_time <= onset_time:
                    logger.warning(
                        f"Heat flow maximum occurs before onset for {sample_short}"
                    )
                    continue

                # Create 6 time points total (onset, 4 intermediate, max_hf)
                time_points = np.linspace(onset_time, max_hf_time, 6)

                # Calculate slopes at each interval
                slopes = []
                slope_times = []

                for i in range(len(time_points) - 1):
                    t1, t2 = time_points[i], time_points[i + 1]

                    # Get data points in this interval
                    interval_data = sample_data[
                        (sample_data[age_col] >= t1) & (sample_data[age_col] <= t2)
                    ]

                    if len(interval_data) < 2:
                        continue

                    # Calculate slope using simple difference
                    x_vals = interval_data[age_col].values
                    y_vals = interval_data[target_col].values

                    slope = (y_vals[-1] - y_vals[0]) / (x_vals[-1] - x_vals[0])
                    slopes.append(slope)
                    slope_times.append((t1 + t2) / 2)  # Midpoint time

                # Include the maximum slope
                max_slope_value = max_slope_row["gradient"].iloc[0]
                slopes.append(max_slope_value)
                slope_times.append(max_slope_time)

                # Calculate average slope
                if slopes:
                    avg_slope = np.mean(slopes)
                    std_slope = np.std(slopes)

                    # Create characteristics dictionary
                    characteristics = {
                        "sample": sample,
                        "sample_short": sample_short,
                        "onset_time_s": onset_time,
                        "max_hf_time_s": max_hf_time,
                        "max_slope_time_s": max_slope_time,
                        "max_slope_value": max_slope_value,
                        "average_slope": avg_slope,
                        "slope_std": std_slope,
                        "n_slopes": len(slopes),
                    }

                    list_of_characteristics.append(characteristics)

            if list_of_characteristics:
                return pd.DataFrame(list_of_characteristics)
            else:
                logger.warning("No average slope characteristics calculated.")
                return pd.DataFrame()

        except Exception as e:
            raise DataProcessingException("get_average_slope", e)


class FlankTangentAnalyzer:
    """Analyzes ascending flank tangents of peaks."""

    def __init__(self, processparams: ProcessingParameters):
        self.processparams = processparams
        self.peak_detector = PeakDetector(processparams)

    def get_ascending_flank_tangent(
        self,
        data: pd.DataFrame,
        target_col: str = "normalized_heat_flow_w_g",
        age_col: str = "time_s",
        #flank_fraction_start: float = 0.2,
        #flank_fraction_end: float = 0.8,
        #window_size: float = 0.1,
        #cutoff_min: Optional[float] = None,
        regex: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Determine tangent to ascending flank of peak by averaging over sections.

        Parameters
        ----------
        data : pd.DataFrame
            Calorimetry data
        target_col : str
            Column containing heat flow data
        age_col : str
            Column containing time data
        flank_fraction_start : float
            Start of flank section as fraction of peak height (0-1)
        flank_fraction_end : float
            End of flank section as fraction of peak height (0-1)
        window_size : float
            Size of averaging window as fraction of flank time range
        cutoff_min : float, optional
            Initial cutoff time in minutes to ignore from analysis. If None,
            uses processparams.cutoff.cutoff_min.
        regex : str, optional
            Regex to filter samples

        Returns
        -------
        pd.DataFrame
            DataFrame with tangent characteristics for each sample
        """
        try:
            from scipy import signal

            cutoff_min = self.processparams.cutoff.cutoff_min if self.processparams.cutoff.cutoff_min else 0
            flank_fraction_start = (
                self.processparams.slope_analysis.flank_fraction_start
            )
            flank_fraction_end = self.processparams.slope_analysis.flank_fraction_end
            window_size = self.processparams.slope_analysis.window_size

            results = []

            for sample, sample_data in SampleIterator.iter_samples(data, regex):
                # Apply cutoff if specified
                # cutoff_time_min = (
                #     cutoff_min
                #     if cutoff_min is not None
                #     else self.processparams.cutoff.cutoff_min
                # )
                if cutoff_min > 0:
                    sample_data = sample_data.query(
                        f"{age_col} >= @cutoff_min * 60"
                    )

                sample_data = sample_data.reset_index(drop=True)

                # Find the main peak
                peaks, _ = signal.find_peaks(
                    sample_data[target_col],
                    prominence=self.processparams.peakdetection.prominence,
                    distance=self.processparams.peakdetection.distance,
                )


                if len(peaks) == 0:
                    logger.warning(f"No peak found in {pathlib.Path(str(sample)).stem}")
                    continue

                # Use the highest peak
                peak_idx = peaks[np.argmax(sample_data.iloc[peaks][target_col])]
                peak_time = sample_data.iloc[peak_idx][age_col]
                peak_value = sample_data.iloc[peak_idx][target_col]

                # Find baseline (minimum before peak)
                baseline_data = sample_data[sample_data[age_col] < peak_time]
                if len(baseline_data) == 0:
                    baseline_value = 0
                else:
                    baseline_value = baseline_data[target_col].min()

                # Define flank region
                flank_height_range = peak_value - baseline_value
                flank_start_value = (
                    baseline_value + flank_fraction_start * flank_height_range
                )
                flank_end_value = (
                    baseline_value + flank_fraction_end * flank_height_range
                )

                # Calculate gradient to ensure we only consider regions with positive slope
                sample_data["gradient"] = np.gradient(
                    sample_data[target_col], sample_data[age_col]
                )

                # Extract ascending flank data - only include points with positive gradient
                flank_data = sample_data[
                    (sample_data[target_col] >= flank_start_value)
                    & (sample_data[target_col] <= flank_end_value)
                    & (sample_data[age_col] <= peak_time)
                    & (sample_data["gradient"] > 0)  # Only positive gradients
                ].copy()

                # If no positive gradient data in initial range, try to find the lowest point with positive gradient
                if len(flank_data) < 3:
                    # Find data points with positive gradient before peak
                    positive_gradient_data = sample_data[
                        (sample_data[age_col] <= peak_time)
                        & (sample_data["gradient"] > 0)
                    ]

                    if len(positive_gradient_data) >= 3:
                        # Adjust flank start to the minimum value with positive gradient
                        min_positive_value = positive_gradient_data[target_col].min()
                        adjusted_flank_start = max(
                            flank_start_value, min_positive_value
                        )

                        flank_data = sample_data[
                            (sample_data[target_col] >= adjusted_flank_start)
                            & (sample_data[target_col] <= flank_end_value)
                            & (sample_data[age_col] <= peak_time)
                            & (sample_data["gradient"] > 0)
                        ].copy()

                        # Update the flank_start_value for recording
                        flank_start_value = adjusted_flank_start

                if len(flank_data) < 3:
                    logger.warning(
                        f"Insufficient flank data in {pathlib.Path(str(sample)).stem}"
                    )
                    continue

                # Calculate moving tangents over windows
                flank_time_range = flank_data[age_col].max() - flank_data[age_col].min()
                window_time = window_size * flank_time_range

                tangent_slopes = []
                tangent_times = []
                tangent_values = []

                # Slide window across flank
                start_time = flank_data[age_col].min()
                end_time = flank_data[age_col].max() - window_time

                step_size = window_time * 1.1  # 10% overlap
                current_time = start_time

                while current_time <= end_time:
                    window_end = current_time + window_time
                    window_data = flank_data[
                        (flank_data[age_col] >= current_time)
                        & (flank_data[age_col] <= window_end)
                    ]

                    if len(window_data) >= 3:
                        # Linear regression for this window
                        x = window_data[age_col].values
                        y = window_data[target_col].values

                        # Use numpy polyfit for linear regression
                        slope, intercept = np.polyfit(x, y, 1)

                        # Only consider positive gradients (ascending flank)
                        if slope > 0:
                            tangent_slopes.append(slope)
                            tangent_times.append(np.mean(x))
                            tangent_values.append(np.mean(y))

                    current_time += step_size

                if not tangent_slopes:
                    logger.warning(
                        f"No valid tangent windows with positive gradients found in {pathlib.Path(str(sample)).stem}"
                    )
                    continue

                # Calculate representative tangent (median to avoid outliers)
                representative_slope = np.median(tangent_slopes)
                representative_time = np.median(tangent_times)
                representative_value = np.median(tangent_values)

                # Calculate tangent line parameters
                # y = mx + b, so b = y - mx
                tangent_intercept = (
                    representative_value - representative_slope * representative_time
                )
                # calculate x intersection
                # y=0, so x = -b/m
                x_intersection = (
                    -tangent_intercept / representative_slope
                    if representative_slope != 0
                    else np.nan
                )

                # Calculate intersection with horizontal line at minimum before tangent_time_s
                data_before_tangent = sample_data[
                    sample_data[age_col] <= representative_time
                ]
                if len(data_before_tangent) > 0:
                    min_value_before_tangent = data_before_tangent[target_col].min()
                    # Intersection: y = min_value = slope * x + intercept
                    # x = (y - intercept) / slope
                    x_intersection_min = (
                        (min_value_before_tangent - tangent_intercept)
                        / representative_slope
                        if representative_slope != 0
                        else np.nan
                    )
                else:
                    min_value_before_tangent = np.nan
                    x_intersection_min = np.nan
                
                # get normalized_heat_j_g at representative_time
                tangent_j_g = np.interp(
                    representative_time,
                    sample_data[age_col],
                    sample_data["normalized_heat_j_g"],
                )

                # get normalized_heat_j_g at peak_time
                peak_j_g = np.interp(
                    peak_time,
                    sample_data[age_col],
                    sample_data["normalized_heat_j_g"],
                )

                # get normalized_heat_j_g at x_intersection
                x_intersection_j_g = np.interp(
                    x_intersection,
                    sample_data[age_col],
                    sample_data["normalized_heat_j_g"],
                ) if not np.isnan(x_intersection) else np.nan

                # get normalized_heat_j_g at x_intersection_min
                x_intersection_dormant_j_g = np.interp(
                    x_intersection_min,
                    sample_data[age_col],
                    sample_data["normalized_heat_j_g"],
                ) if not np.isnan(x_intersection_min) else np.nan

                result = {
                    "sample": sample,
                    "sample_short": pathlib.Path(str(sample)).stem,
                    "peak_time_s": peak_time,
                    "peak_value": peak_value,
                    "peak_j_g": peak_j_g,
                    "tangent_slope": representative_slope,
                    "tangent_time_s": representative_time,
                    "tangent_value": representative_value,
                    "tangent_intercept": tangent_intercept,
                    "tangent_j_g": tangent_j_g,
                    "flank_start_value": flank_start_value,
                    "flank_end_value": flank_end_value,
                    "n_windows": len(tangent_slopes),
                    "slope_std": np.std(tangent_slopes),
                    "x_intersection": x_intersection,
                    "min_value_before_tangent": min_value_before_tangent,
                    "x_intersection_dormant": x_intersection_min,
                    "x_intersection_dormant_j_g": x_intersection_dormant_j_g,
                    "x_intersection_j_g": x_intersection_j_g,
                }

                results.append(result)

            return pd.DataFrame(results)

        except Exception as e:
            raise DataProcessingException("get_ascending_flank_tangent", e)
