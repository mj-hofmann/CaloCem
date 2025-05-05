import os
import pytest
import tempfile
import pandas as pd
from pathlib import Path
from unittest.mock import patch, MagicMock

from calocem.tacalorimetry import Measurement, ProcessingParameters


class TestMeasurement:
    
    def test_init_empty(self):
        """Test initialization without folder."""
        measurement = Measurement()
        assert measurement._data.empty
        assert measurement._info.empty
        
    @patch('os.listdir')
    @patch('calocem.tacalorimetry.Measurement._read_csv_data')
    def test_init_with_folder_new_code(self, mock_read_csv, mock_listdir):
        """Test initialization with folder and new_code=True."""
        # Mock setup
        mock_listdir.return_value = ['sample1.csv', 'sample2.csv', 'other.txt']
        mock_df = pd.DataFrame({'time_s': [1, 2], 'heat_flow_w': [0.1, 0.2]})
        mock_read_csv.return_value = mock_df
        
        # Initialize with new_code=True
        measurement = Measurement(folder="/fake/path", new_code=True)
        
        # Check that _read_csv_data was called for each CSV file
        assert mock_read_csv.call_count == 2
        assert not measurement._data.empty
        
    @patch('os.listdir')
    @patch('calocem.tacalorimetry.Measurement._read_calo_data_csv')
    @patch('calocem.tacalorimetry.Measurement._read_calo_info_csv')
    def test_init_with_folder_old_code(self, mock_read_info, mock_read_data, mock_listdir):
        """Test initialization with folder and new_code=False."""
        # Mock setup
        mock_listdir.return_value = ['sample1.csv', 'sample2.csv']
        mock_df = pd.DataFrame({'time_s': [1, 2], 'heat_flow_w': [0.1, 0.2]})
        mock_read_data.return_value = mock_df
        mock_read_info.return_value = pd.DataFrame({'parameter': ['test'], 'value': ['value']})
        
        # Initialize with new_code=False
        measurement = Measurement(folder="/fake/path", new_code=False)
        
        # Check that _read_calo_data_csv and _read_calo_info_csv were called for each CSV file
        assert mock_read_data.call_count == 2
        assert mock_read_info.call_count == 2
        assert not measurement._data.empty
        assert not measurement._info.empty
        
    @patch('os.listdir')
    @patch('calocem.tacalorimetry.Measurement._read_calo_data_xls')
    @patch('calocem.tacalorimetry.Measurement._read_calo_info_xls')
    def test_init_with_xls_files(self, mock_read_info, mock_read_data, mock_listdir):
        """Test initialization with XLS files."""
        # Mock setup
        mock_listdir.return_value = ['sample1.xls', 'sample2.xls']
        mock_df = pd.DataFrame({'time_s': [1, 2], 'heat_flow_w': [0.1, 0.2]})
        mock_read_data.return_value = mock_df
        mock_read_info.return_value = pd.DataFrame({'parameter': ['test'], 'value': ['value']})
        
        # Initialize with new_code=False for XLS files
        measurement = Measurement(folder="/fake/path", new_code=False)
        
        # Check that proper methods were called for XLS files
        assert mock_read_data.call_count == 2
        assert mock_read_info.call_count == 2
        assert not measurement._data.empty
        assert not measurement._info.empty
    
    @patch('os.listdir')
    def test_init_with_regex_filter(self, mock_listdir):
        """Test initialization with regex filter."""
        # Mock setup
        mock_listdir.return_value = ['sample1.csv', 'test2.csv', 'other.txt']
        
        with patch.object(Measurement, '_read_csv_data') as mock_read:
            mock_read.return_value = pd.DataFrame({'time_s': [1], 'heat_flow_w': [0.1]})
            
            # Initialize with regex to filter only 'sample' files
            measurement = Measurement(folder="/fake/path", new_code=True, regex=r"sample.*\.csv")
            
            # Check that _read_csv_data was called only for matching files
            assert mock_read.call_count == 1
    
    @patch('calocem.tacalorimetry.Measurement._get_data_and_parameters_from_folder')
    @patch('calocem.tacalorimetry.Measurement._auto_clean_data')
    def test_init_with_auto_clean(self, mock_auto_clean, mock_get_data):
        """Test initialization with auto_clean=True."""
        # Initialize with auto_clean=True
        measurement = Measurement(folder="/fake/path", auto_clean=True)
        
        # Check that _auto_clean_data was called
        mock_auto_clean.assert_called_once()
    
    @patch('os.listdir')
    @patch('calocem.tacalorimetry.Measurement._get_data_and_parameters_from_pickle')
    def test_init_with_cold_start_false(self, mock_get_from_pickle, mock_listdir):
        """Test initialization with cold_start=False."""
        # Initialize with cold_start=False
        measurement = Measurement(folder="/fake/path", cold_start=False)
        
        # Check that _get_data_and_parameters_from_pickle was called
        mock_get_from_pickle.assert_called_once()
        
    def test_init_with_custom_processparams(self):
        """Test initialization with custom ProcessingParameters."""
        custom_params = ProcessingParameters()
        custom_params.cutoff.cutoff_min = 15  # Change a parameter
        
        measurement = Measurement(processparams=custom_params)
        
        # Check that custom parameters were used
        assert measurement.processparams.cutoff.cutoff_min == 15
