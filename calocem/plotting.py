"""
Plotting functionality for calorimetry data visualization.
"""

import logging
import pathlib
from typing import Optional

import matplotlib.axes
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .data_processing import SampleIterator
from .exceptions import DataProcessingException

logger = logging.getLogger(__name__)


class SimplePlotter:
    """Simplified plotting functionality to avoid complex type issues."""

    def plot_data(
        self,
        data: pd.DataFrame,
        t_unit: str = "h",
        y: str = "normalized_heat_flow_w_g",
        y_unit_milli: bool = True,
        regex: Optional[str] = None,
        show_info: bool = True,
        ax: Optional[matplotlib.axes.Axes] = None,
    ):
        """Plot the calorimetry data."""
        try:
            # Setup unit conversions
            unit_conversions = {
                "s": (1.0, "Time [s]"),
                "min": (1 / 60, "Time [min]"),
                "h": (1 / 3600, "Time [h]"),
                "d": (1 / (24 * 3600), "Time [d]"),
            }

            y_configs = {
                "normalized_heat_flow_w_g": "Normalized Heat Flow / [W/g]",
                "heat_flow_w": "Heat Flow / [W]",
                "normalized_heat_j_g": "Normalized Heat / [J/g]",
                "heat_j": "Heat / [J]",
            }

            x_factor, x_label = unit_conversions.get(t_unit, (1.0, "Time [s]"))
            y_label = y_configs.get(y, y)
            y_factor = 1000 if y_unit_milli else 1

            if y_unit_milli:
                y_label = y_label.replace("[", "[m")

            # Create figure if no axis provided
            if ax is None:
                fig, ax = plt.subplots(figsize=(10, 6))

            # Plot each sample
            for sample, sample_data in SampleIterator.iter_samples(data, regex):
                if show_info:
                    print(f"Plotting {pathlib.Path(str(sample)).stem}")

                # Apply unit conversions
                plot_data = sample_data.copy()
                plot_data["time_s"] = plot_data["time_s"] * x_factor

                # Convert heat columns
                heat_cols = [col for col in plot_data.columns if "heat" in col]
                plot_data[heat_cols] = plot_data[heat_cols] * y_factor

                # Plot
                ax.plot(
                    plot_data["time_s"],
                    plot_data[y],
                    label=pathlib.Path(str(sample)).stem,
                )

            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)
            ax.legend()
            ax.grid(True, alpha=0.3)

            return ax

        except Exception as e:
            raise DataProcessingException("plot_data", e)

    def plot_peaks(
        self,
        data: pd.DataFrame,
        peaks: np.ndarray,
        sample: str,
        ax: Optional[matplotlib.axes.Axes] = None,
        age_col: str = "time_s",
        target_col: str = "normalized_heat_flow_w_g",
    ):
        """Plot detected peaks."""
        try:
            if ax is None:
                fig, ax = plt.subplots(figsize=(10, 6))

            # Plot main data
            ax.plot(data[age_col], data[target_col], label=sample)

            # Mark peaks
            if len(peaks) > 0:
                ax.plot(
                    data[age_col].iloc[peaks],
                    data[target_col].iloc[peaks],
                    "rx",
                    markersize=10,
                    label="Peaks",
                )

                # Add vertical lines
                for peak_idx in peaks:
                    ax.axvline(
                        data[age_col].iloc[peak_idx],
                        color="red",
                        alpha=0.5,
                        linestyle="--",
                    )

            ax.set_xlabel(age_col)
            ax.set_ylabel(target_col)
            ax.set_title(f"Peak Detection - {sample}")
            ax.legend()
            ax.grid(True, alpha=0.3)

            return ax

        except Exception as e:
            raise DataProcessingException("plot_peaks", e)

    def plot_slopes(
        self,
        data: pd.DataFrame,
        characteristics: pd.DataFrame,
        sample: str,
        ax: Optional[matplotlib.axes.Axes] = None,
        age_col: str = "time_s",
        target_col: str = "normalized_heat_flow_w_g",
        tangent_length_factor: float = 0.3,
    ):
        """
        Plot slope analysis with tangent lines.

        Parameters
        ----------
        data : pd.DataFrame
            Sample data containing time and heat flow columns
        characteristics : pd.DataFrame
            DataFrame with slope characteristics containing at least:
            - age_col: time at maximum slope
            - target_col: heat flow value at maximum slope
            - 'gradient': slope value (if available)
        sample : str
            Sample name for labeling
        ax : matplotlib.axes.Axes, optional
            Matplotlib axes to plot on
        age_col : str
            Column name for time data
        target_col : str
            Column name for heat flow data
        tangent_length_factor : float
            Factor determining tangent line length relative to data range
        """
        try:
            if ax is None:
                fig, ax = plt.subplots(figsize=(10, 6))

            # Plot main data
            ax.plot(data[age_col], data[target_col], label=sample)

            # Calculate data range for tangent line length
            x_range = data[age_col].max() - data[age_col].min()
            tangent_half_length = x_range * tangent_length_factor / 2

            # Mark maximum slope points and plot tangent lines
            for i, row in characteristics.iterrows():
                slope_time = row[age_col]
                slope_value = row[target_col]

                # Plot vertical line at maximum slope position
                ax.axvline(
                    slope_time,
                    color="gray",
                    linestyle="--",
                    alpha=0.5,
                    label="Max Slope Position" if i == 0 else "",
                )

                # Plot tangent line if gradient information is available
                if "gradient" in row and not pd.isna(row["gradient"]):
                    gradient = row["gradient"]

                    # Calculate tangent line endpoints
                    x_start = slope_time - tangent_half_length
                    x_end = slope_time + tangent_half_length

                    # Ensure tangent line stays within data bounds
                    x_start = max(x_start, data[age_col].min())
                    x_end = min(x_end, data[age_col].max())

                    # Calculate y values using point-slope form: y - y1 = m(x - x1)
                    y_start = slope_value + gradient * (x_start - slope_time)
                    y_end = slope_value + gradient * (x_end - slope_time)

                    # Plot tangent line
                    ax.plot(
                        [x_start, x_end],
                        [y_start, y_end],
                        color="orange",
                        linewidth=1.5,
                        alpha=0.7,
                        label="Tangent (Max Slope)" if i == 0 else "",
                    )

                    # Add annotation with slope value
                    ax.annotate(
                        f"Max Slope\n{slope_time:.1f}s\nGradient: {gradient:.2e}",
                        xy=(slope_time, slope_value),
                        xytext=(20, 20),
                        textcoords="offset points",
                        bbox=dict(
                            boxstyle="round,pad=0.1", facecolor="white", alpha=0.4
                        ),
                        arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0"),
                    )
                else:
                    # Fallback annotation without gradient info
                    ax.annotate(
                        f"Max Slope\n{slope_time:.1f}s",
                        xy=(slope_time, slope_value),
                        xytext=(10, 10),
                        textcoords="offset points",
                        bbox=dict(
                            boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7
                        ),
                        arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0"),
                    )

            ax.set_xlabel(age_col)
            ax.set_ylabel(target_col)
            ax.set_title(f"Slope Analysis - {sample}")
            ax.legend()
            ax.grid(True, alpha=0.3)

            return ax

        except Exception as e:
            raise DataProcessingException("plot_slopes", e)

    def plot_tangent_analysis(
        self,
        data: pd.DataFrame,
        sample: str,
        ax: Optional[matplotlib.axes.Axes] = None,
        age_col: str = "time_s",
        target_col: str = "normalized_heat_flow_w_g",
        cutoff_time_min: Optional[float] = None,
        analysis_type: str = "flank_tangent",
        # Flank tangent specific parameters
        tangent_results: Optional[pd.DataFrame] = None,
        # Onset intersection specific parameters
        max_slopes: Optional[pd.DataFrame] = None,
        dormant_hfs: Optional[pd.DataFrame] = None,
        onsets: Optional[pd.DataFrame] = None,
        intersection: str = "dormant_hf",
        xunit: str = "s",
        # Common styling parameters
        figsize: tuple = (10, 6),
    ):
        """
        Unified plotting method for tangent-based analysis results.

        This method can handle both flank tangent analysis and onset intersection
        analysis, providing a consistent interface and reducing code duplication.

        Parameters
        ----------
        data : pd.DataFrame
            Sample data containing time and heat flow columns
        sample : str
            Sample name for labeling
        ax : matplotlib.axes.Axes, optional
            Matplotlib axes to plot on
        age_col : str
            Column name for time data
        target_col : str
            Column name for heat flow data
        cutoff_time_min : float, optional
            Cutoff time in minutes to show as vertical line if data was filtered
        analysis_type : str
            Type of analysis: 'flank_tangent' or 'onset_intersection'
        tangent_results : pd.DataFrame, optional
            For flank_tangent: DataFrame with tangent characteristics
        max_slopes : pd.DataFrame, optional
            For onset_intersection: DataFrame with maximum slope characteristics
        dormant_hfs : pd.DataFrame, optional
            For onset_intersection: DataFrame with dormant heat flow data
        onsets : pd.DataFrame, optional
            For onset_intersection: DataFrame with calculated onset times
        intersection : str
            For onset_intersection: Type of intersection ('dormant_hf' or 'abscissa')
        xunit : str
            Time unit for plotting
        figsize : tuple
            Figure size for the plot
        """
        try:
            if ax is None:
                fig, ax = plt.subplots(figsize=figsize)

            # Plot main data
            ax.plot(data[age_col], data[target_col], "b-", alpha=0.7, label="Data")

            # Show cutoff line if data was filtered
            if cutoff_time_min is not None:
                cutoff_seconds = cutoff_time_min * 60
                if cutoff_seconds > data[age_col].min():
                    ax.axvline(
                        cutoff_seconds,
                        color="gray",
                        linestyle="-.",
                        alpha=0.8,
                        label=f"Cutoff: {cutoff_time_min:.0f} min",
                    )

            if analysis_type == "flank_tangent":
                self._plot_flank_tangent_elements(
                    ax, data, tangent_results, sample, age_col, target_col
                )
            elif analysis_type == "onset_intersection":
                self._plot_onset_intersection_elements(
                    ax,
                    data,
                    max_slopes,
                    dormant_hfs,
                    onsets,
                    sample,
                    age_col,
                    target_col,
                    intersection,
                )
            else:
                raise ValueError(f"Unknown analysis_type: {analysis_type}")

            # Common styling
            ax.set_xlabel(f"{age_col.replace('_', ' ').title()}")
            ax.set_ylabel(f"{target_col.replace('_', ' ').title()}")
            ax.legend(loc="best")
            ax.grid(True, alpha=0.3)
            ax.set_ylim(bottom=0)

            return ax

        except Exception as e:
            raise DataProcessingException("plot_tangent_analysis", e)

    def _plot_flank_tangent_elements(
        self,
        ax: matplotlib.axes.Axes,
        data: pd.DataFrame,
        tangent_results: Optional[pd.DataFrame],
        sample: str,
        age_col: str,
        target_col: str,
    ):
        """Plot elements specific to flank tangent analysis."""
        if tangent_results is None:
            raise ValueError("tangent_results required for flank_tangent analysis")

        for _, result in tangent_results.iterrows():
            peak_time = result["peak_time_s"]
            peak_value = result["peak_value"]
            tangent_slope = result["tangent_slope"]
            tangent_intercept = result["tangent_intercept"]
            flank_start_value = result["flank_start_value"]
            flank_end_value = result["flank_end_value"]
            x_intersection = result.get("x_intersection", np.nan)

            # Mark peak
            ax.plot(peak_time, peak_value, "ro", markersize=8, label="Peak Maximum")

            # Mark flank region
            ax.axhline(
                flank_start_value,
                color="green",
                linestyle=":",
                alpha=0.7,
                label="Flank Region",
            )
            ax.axhline(flank_end_value, color="green", linestyle=":", alpha=0.7)

            # Highlight flank region
            flank_data = data[
                (data[target_col] >= flank_start_value)
                & (data[target_col] <= flank_end_value)
                & (data[age_col] <= peak_time)
            ]
            if not flank_data.empty:
                ax.fill_between(
                    flank_data[age_col],
                    flank_start_value,
                    flank_end_value,
                    alpha=0.2,
                    color="green",
                )

            # Plot tangent line
            if not np.isnan(x_intersection):
                x_tangent = np.linspace(x_intersection, peak_time, 100)
            else:
                x_min = (
                    flank_data[age_col].min()
                    if not flank_data.empty
                    else data[age_col].min()
                )
                x_tangent = np.linspace(x_min, peak_time, 100)

            y_tangent = tangent_slope * x_tangent + tangent_intercept
            ax.plot(
                x_tangent,
                y_tangent,
                "r-",
                linewidth=2,
                alpha=0.8,
                label="Flank Tangent",
            )

            # Mark x-intersection if available
            if not np.isnan(x_intersection) and x_intersection > data[age_col].min():
                ax.axvline(
                    x_intersection,
                    color="orange",
                    linestyle="-.",
                    alpha=0.7,
                    label=f"X-intercept: {x_intersection:.0f}s",
                )
                ax.plot(x_intersection, 0, "o", color="orange", markersize=8)

            # Mark minimum intersection if available
            min_value_before_tangent = result.get("min_value_before_tangent", np.nan)
            x_intersection_min = result.get("x_intersection_min", np.nan)

            if not np.isnan(min_value_before_tangent) and not np.isnan(
                x_intersection_min
            ):
                ax.axhline(
                    min_value_before_tangent,
                    color="purple",
                    linestyle="-.",
                    alpha=0.7,
                    label=f"Min before tangent: {min_value_before_tangent:.2e}",
                )
                ax.axvline(
                    x_intersection_min,
                    color="purple",
                    linestyle=":",
                    alpha=0.7,
                    label=f"Min intersection: {x_intersection_min:.0f}s",
                )
                ax.plot(
                    x_intersection_min,
                    min_value_before_tangent,
                    "s",
                    color="purple",
                    markersize=4,
                    label="Tangent-Min Intersection",
                )

            # Add annotation with slope information
            mid_x = (
                x_intersection
                if not np.isnan(x_intersection)
                else flank_data[age_col].min()
            ) + (
                peak_time
                - (
                    x_intersection
                    if not np.isnan(x_intersection)
                    else flank_data[age_col].min()
                )
            ) / 2
            mid_y = tangent_slope * mid_x + tangent_intercept

            ax.annotate(
                f"Tangent Slope: {tangent_slope:.2e}",
                xy=(mid_x, mid_y),
                xytext=(10, 10),
                textcoords="offset points",
                fontsize=10,
                color="red",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0"),
            )

        ax.set_title(f"Ascending Flank Tangent Analysis - {sample}")

    def _plot_onset_intersection_elements(
        self,
        ax: matplotlib.axes.Axes,
        data: pd.DataFrame,
        max_slopes: Optional[pd.DataFrame],
        dormant_hfs: Optional[pd.DataFrame],
        onsets: Optional[pd.DataFrame],
        sample: str,
        age_col: str,
        target_col: str,
        intersection: str,
    ):
        """Plot elements specific to onset intersection analysis."""
        if max_slopes is None or onsets is None:
            raise ValueError(
                "max_slopes and onsets required for onset_intersection analysis"
            )

        # Get sample-specific data
        sample_max_slope = max_slopes[max_slopes["sample_short"] == sample]
        sample_dormant = (
            dormant_hfs[dormant_hfs["sample_short"] == sample]
            if dormant_hfs is not None
            else pd.DataFrame()
        )
        sample_onset = onsets[onsets["sample"] == sample]

        if sample_max_slope.empty or sample_onset.empty:
            ax.set_title(f"No data available for sample: {sample}")
            return

        # Extract slope characteristics
        slope_time = sample_max_slope[age_col].iloc[0]
        slope_value = sample_max_slope[target_col].iloc[0]
        gradient = sample_max_slope["gradient"].iloc[0]
        onset_time = sample_onset["onset_time_s"].iloc[0]

        # Plot maximum slope point
        ax.plot(
            slope_time,
            slope_value,
            "go",
            markersize=10,
            label="Maximum Slope",
            zorder=5,
        )
        ax.axvline(
            slope_time,
            color="green",
            linestyle="--",
            alpha=0.7,
            label="Max Slope Time",
        )

        # Calculate and plot tangent line
        tangent_intercept = slope_value - gradient * slope_time
        x_start = min(onset_time, data[age_col].min())
        x_end = max(slope_time * 1.2, data[age_col].max() * 0.8)
        x_tangent = np.linspace(x_start, x_end, 100)
        y_tangent = gradient * x_tangent + tangent_intercept

        ax.plot(
            x_tangent,
            y_tangent,
            "r-",
            linewidth=2,
            alpha=0.8,
            label="Maximum Slope Tangent",
        )

        # Plot intersection line and onset point
        if intersection == "dormant_hf" and not sample_dormant.empty:
            dormant_value = sample_dormant["normalized_heat_flow_w_g"].iloc[0]
            ax.axhline(
                dormant_value,
                color="orange",
                linestyle=":",
                alpha=0.8,
                label=f"Dormant Heat Flow: {dormant_value:.2e}",
            )
            ax.plot(
                onset_time,
                dormant_value,
                "ro",
                markersize=10,
                label=f"Onset (Dormant HF): {onset_time:.0f}s",
                zorder=5,
            )
            onset_y_pos = dormant_value
        else:
            ax.axhline(
                0, color="orange", linestyle=":", alpha=0.8, label="Abscissa (y=0)"
            )
            ax.plot(
                onset_time,
                0,
                "ro",
                markersize=10,
                label=f"Onset (Abscissa): {onset_time:.0f}s",
                zorder=5,
            )
            onset_y_pos = 0

        # Mark onset time with vertical line
        ax.axvline(
            onset_time,
            color="red",
            linestyle="-",
            alpha=0.8,
            linewidth=2,
            label=f"Onset Time: {onset_time:.0f}s",
        )

        # Add annotations
        ax.annotate(
            f"Onset: {onset_time:.0f}s\n({onset_time/60:.1f} min)",
            xy=(onset_time, onset_y_pos),
            xytext=(20, 20),
            textcoords="offset points",
            fontsize=10,
            color="red",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.8),
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0"),
        )

        mid_x = (onset_time + slope_time) / 2
        mid_y = gradient * mid_x + tangent_intercept
        ax.annotate(
            f"Gradient: {gradient:.2e}",
            xy=(mid_x, mid_y),
            xytext=(10, -20),
            textcoords="offset points",
            fontsize=9,
            color="red",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8),
        )

        intersection_type = (
            "Dormant Heat Flow" if intersection == "dormant_hf" else "Abscissa"
        )
        ax.set_title(
            f"Peak Onset via Max Slope - {intersection_type} Intersection\nSample: {sample}"
        )

        return ax

    def plot_flank_tangent(
        self,
        data: pd.DataFrame,
        tangent_results: pd.DataFrame,
        sample: str,
        ax: Optional[matplotlib.axes.Axes] = None,
        age_col: str = "time_s",
        target_col: str = "normalized_heat_flow_w_g",
        cutoff_time_min: Optional[float] = None,
    ):
        """
        Plot ascending flank tangent analysis results.

        This is a wrapper around plot_tangent_analysis for backward compatibility.
        """
        return self.plot_tangent_analysis(
            data=data,
            sample=sample,
            ax=ax,
            age_col=age_col,
            target_col=target_col,
            cutoff_time_min=cutoff_time_min,
            analysis_type="flank_tangent",
            tangent_results=tangent_results,
            figsize=(7, 5),
        )

    def plot_onset_intersections(
        self,
        data: pd.DataFrame,
        max_slopes: pd.DataFrame,
        dormant_hfs: pd.DataFrame,
        onsets: pd.DataFrame,
        sample: str,
        ax: Optional[matplotlib.axes.Axes] = None,
        age_col: str = "time_s",
        target_col: str = "normalized_heat_flow_w_g",
        intersection: str = "dormant_hf",
        xunit: str = "s",
        cutoff_time_min: Optional[float] = None,
    ):
        """
        Plot onset intersections via maximum slope method.

        This is a wrapper around plot_tangent_analysis for backward compatibility.
        """
        return self.plot_tangent_analysis(
            data=data,
            sample=sample,
            ax=ax,
            age_col=age_col,
            target_col=target_col,
            cutoff_time_min=cutoff_time_min,
            analysis_type="onset_intersection",
            max_slopes=max_slopes,
            dormant_hfs=dormant_hfs,
            onsets=onsets,
            intersection=intersection,
            xunit=xunit,
            figsize=(12, 8),
        )
