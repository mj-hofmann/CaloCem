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
    ):
        """Plot slope analysis."""
        try:
            if ax is None:
                fig, ax = plt.subplots(figsize=(10, 6))

            # Plot main data
            ax.plot(data[age_col], data[target_col], label=sample)

            # Mark maximum slope points
            for _, row in characteristics.iterrows():
                ax.axvline(
                    row[age_col],
                    color="green",
                    linestyle="--",
                    alpha=0.7,
                    label="Max Slope",
                )

                # Add annotation
                ax.annotate(
                    f"Max Slope\n{row[age_col]:.1f}s",
                    xy=(row[age_col], row[target_col]),
                    xytext=(10, 10),
                    textcoords="offset points",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7),
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
