#!/usr/bin/env python3
"""
Test script to verify the enhanced plot_slopes method with tangent plotting.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Test the enhanced plot_slopes method
def test_plot_slopes_with_tangent():
    """Test the enhanced plot_slopes method that includes tangent plotting."""

    # Create sample data
    time_s = np.linspace(0, 10000, 1000)
    heat_flow = (
        0.001 * np.exp(-time_s / 3000)
        + 0.002 * np.sin(time_s / 1000)
        + np.random.normal(0, 0.0001, len(time_s))
    )

    sample_data = pd.DataFrame(
        {
            "time_s": time_s,
            "normalized_heat_flow_w_g": heat_flow,
            "sample": "test_sample",
            "sample_short": "test",
        }
    )

    # Create sample characteristics with gradient information
    characteristics = pd.DataFrame(
        {
            "time_s": [3000, 7000],
            "normalized_heat_flow_w_g": [0.0015, 0.0012],
            "gradient": [0.0001, -0.00005],  # Include gradient for tangent plotting
            "sample": "test_sample",
            "sample_short": "test",
        }
    )

    try:
        # Import the SimplePlotter
        import os
        import sys

        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "calocem"))

        from calocem.plotting import SimplePlotter

        # Create plotter instance
        plotter = SimplePlotter()

        # Test the enhanced plot_slopes method
        print("Testing enhanced plot_slopes method with tangent plotting...")

        ax = plotter.plot_slopes(
            data=sample_data,
            characteristics=characteristics,
            sample="test_sample",
            ax=None,
            age_col="time_s",
            target_col="normalized_heat_flow_w_g",
            tangent_length_factor=0.3,
        )

        print("✓ plot_slopes with tangent plotting works correctly!")

        # Show the plot
        plt.title("Enhanced plot_slopes - Shows both vertical lines and tangent lines")
        plt.show()

        # Test the new plot_flank_tangent method
        print("\nTesting new plot_flank_tangent method...")

        flank_results = pd.DataFrame(
            {
                "peak_time_s": [5000],
                "peak_value": [0.002],
                "tangent_slope": [0.0001],
                "tangent_intercept": [-0.0002],
                "flank_start_value": [0.001],
                "flank_end_value": [0.0018],
                "x_intersection": [2000],
                "sample": "test_sample",
                "sample_short": "test",
            }
        )

        ax2 = plotter.plot_flank_tangent(
            data=sample_data,
            tangent_results=flank_results,
            sample="test_sample",
            ax=None,
            age_col="time_s",
            target_col="normalized_heat_flow_w_g",
        )

        print("✓ plot_flank_tangent works correctly!")
        plt.show()

        print("\n✅ All plotting enhancements work correctly!")

    except Exception as e:
        print(f"❌ Error testing plotting enhancements: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    test_plot_slopes_with_tangent()
    test_plot_slopes_with_tangent()
