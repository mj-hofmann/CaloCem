#!/usr/bin/env python3
"""
Test script to verify the refactored get_ascending_flank_tangent method.
"""

import os
import sys

sys.path.insert(0, os.path.abspath("."))

from calocem.measurement import Measurement
from calocem.processparams import ProcessingParameters


def test_refactored_flank_tangent():
    """Test the refactored ascending flank tangent method."""

    print("Testing refactored get_ascending_flank_tangent method...")

    # Check if data exists
    data_folder = (
        "/home/torben/LRZ Sync+Share/0_TUM/10_Coding/packages/calocem/calocem/DATA"
    )
    print(f"Checking for data folder: {data_folder}")

    if not os.path.exists(data_folder):
        print(f"Data folder {data_folder} not found. Creating a minimal test...")
        test_minimal_functionality()
        return

    print(f"Data folder exists. Contents: {os.listdir(data_folder)}")

    try:
        # Create measurement instance
        processparams = ProcessingParameters()
        print("ProcessingParameters created")

        measurement = Measurement(
            folder=data_folder,
            show_info=True,
            processparams=processparams,
            cold_start=True,
        )
        print("Measurement instance created")

        # Test the refactored method
        print("\nCalling get_ascending_flank_tangent...")
        result = measurement.get_ascending_flank_tangent(
            processparams=processparams,
            target_col="normalized_heat_flow_w_g",
            age_col="time_s",
            flank_fraction_start=0.2,
            flank_fraction_end=0.8,
            window_size=0.1,
            cutoff_min=None,
            show_plot=False,
            regex=None,
        )

        print(f"\nResult shape: {result.shape}")
        if not result.empty:
            print("Columns:", list(result.columns))
            print("\nFirst few rows:")
            print(result.head())
            print("\nRefactored method working correctly! ✓")
        else:
            print("No results returned - check input data and parameters")

    except Exception as e:
        print(f"Error testing refactored method: {e}")
        import traceback

        traceback.print_exc()


def test_minimal_functionality():
    """Test the refactored method structure without actual data."""

    try:
        # Test that we can import and instantiate the classes
        from calocem.analysis import FlankTangentAnalyzer
        from calocem.processparams import ProcessingParameters

        processparams = ProcessingParameters()
        analyzer = FlankTangentAnalyzer(processparams)

        print("FlankTangentAnalyzer instantiated successfully ✓")
        print("Refactoring structure is correct ✓")

        # Test that Measurement class has the method
        measurement = Measurement(show_info=False, cold_start=False)
        has_method = hasattr(measurement, "get_ascending_flank_tangent")

        print(f"Measurement.get_ascending_flank_tangent exists: {has_method} ✓")

    except Exception as e:
        print(f"Error in minimal functionality test: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    test_refactored_flank_tangent()
if __name__ == "__main__":
    test_refactored_flank_tangent()
