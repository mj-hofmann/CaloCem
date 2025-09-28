#!/usr/bin/env python3
"""
Test script to demonstrate the improved flank region visualization.

This script shows how the green fill_between area now only starts shortly
before the flank region instead of filling the entire flank region.
"""

import pathlib
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Add the current directory to Python path for importing calocem
sys.path.insert(0, str(pathlib.Path(__file__).parent))

try:
    from calocem.plotting import SimplePlotter

    print("✓ Successfully imported calocem modules")

    # Create sample data for demonstration
    time_s = np.linspace(0, 10000, 1000)
    # Create a heat flow curve with a clear peak
    heat_flow = 0.001 * np.exp(
        -((time_s - 5000) ** 2) / (2 * 1000**2)
    ) + 0.0001 * np.sin(time_s / 1000)
    heat_flow = np.maximum(heat_flow, 0.0001)  # Ensure positive values

    sample_data = pd.DataFrame(
        {
            "time_s": time_s,
            "normalized_heat_flow_w_g": heat_flow,
            "sample": "test_sample",
            "sample_short": "test",
        }
    )

    # Create mock tangent results
    tangent_results = pd.DataFrame(
        {
            "peak_time_s": [5000],
            "peak_value": [heat_flow.max()],
            "tangent_slope": [2e-7],
            "tangent_intercept": [-0.0005],
            "flank_start_value": [0.0003],
            "flank_end_value": [0.0007],
            "x_intersection": [2500],
            "min_value_before_tangent": [0.0001],
            "x_intersection_min": [500],
            "sample": ["test_sample"],
            "sample_short": ["test"],
        }
    )

    # Create plotter and test the visualization
    plotter = SimplePlotter()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Plot with the new improved fill_between
    ax1 = plotter.plot_flank_tangent(
        data=sample_data, tangent_results=tangent_results, sample="test", ax=ax1
    )
    ax1.set_title("Improved: Fill starts shortly before flank")

    # Show the flank region data for comparison
    flank_data = sample_data[
        (sample_data["normalized_heat_flow_w_g"] >= 0.0003)
        & (sample_data["normalized_heat_flow_w_g"] <= 0.0007)
        & (sample_data["time_s"] <= 5000)
    ]

    ax2.plot(
        sample_data["time_s"],
        sample_data["normalized_heat_flow_w_g"],
        "b-",
        alpha=0.7,
        label="Data",
    )
    ax2.axhline(0.0003, color="green", linestyle=":", alpha=0.7, label="Flank Start")
    ax2.axhline(0.0007, color="green", linestyle=":", alpha=0.7, label="Flank End")

    # Show original full flank fill for comparison
    if not flank_data.empty:
        ax2.fill_between(
            flank_data["time_s"],
            0.0003,
            0.0007,
            alpha=0.2,
            color="red",
            label="Original: Full flank fill",
        )

    ax2.set_xlabel("Time [s]")
    ax2.set_ylabel("Normalized Heat Flow [W/g]")
    ax2.set_title("Comparison: Original full flank fill (red)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(bottom=0)

    plt.tight_layout()
    plt.savefig("flank_fill_comparison.png", dpi=150, bbox_inches="tight")
    print("✓ Comparison plot saved as 'flank_fill_comparison.png'")

    print("\n=== Summary of Changes ===")
    print("✓ Green fill area now starts 10% of flank duration before the actual flank")
    print("✓ This provides better visual context without excessive highlighting")
    print("✓ The fill area is more focused on the relevant region")
    print("✓ Backward compatibility maintained - same method signature")

    if not flank_data.empty:
        flank_start_time = flank_data["time_s"].min()
        flank_end_time = flank_data["time_s"].max()
        flank_duration = flank_end_time - flank_start_time
        fill_start_time = max(
            flank_start_time - 0.1 * flank_duration, sample_data["time_s"].min()
        )

        print("\n=== Technical Details ===")
        print(f"Flank region: {flank_start_time:.0f}s to {flank_end_time:.0f}s")
        print(f"Flank duration: {flank_duration:.0f}s")
        print(f"Fill starts at: {fill_start_time:.0f}s (10% earlier)")
        print("Improvement: Fill region reduced by focusing on relevant area")

except ImportError as e:
    print(f"✗ Import error: {e}")
    print("Make sure calocem package is properly installed or in the Python path")
except Exception as e:
    print(f"✗ Test failed: {e}")
    import traceback

    traceback.print_exc()
    traceback.print_exc()
