"""
Custom exceptions for the CaloCem package.
"""


class AutoCleanException(Exception):
    """Raised when automatic data cleaning fails."""

    def __init__(self):
        message = "auto_clean failed. Consider switching to turn this option off."
        super().__init__(message)


class ColdStartException(Exception):
    """Raised when cold start initialization fails."""

    def __init__(self):
        message = "cold_start failed. Consider switching to cold_start=True."
        super().__init__(message)


class AddMetaDataSourceException(Exception):
    """Raised when metadata source addition fails."""

    def __init__(self, list_of_possible_ids):
        message = "The specified id column is not available in the declared file. Please use one of"
        for option in list_of_possible_ids:
            message += f"\n  - {option}"
        super().__init__(message)


class FileReadingException(Exception):
    """Raised when file reading fails."""

    def __init__(self, filename, original_error=None):
        message = f"Failed to read file: {filename}"
        if original_error:
            message += f"\nOriginal error: {original_error}"
        super().__init__(message)


class DataProcessingException(Exception):
    """Raised when data processing fails."""

    def __init__(self, operation, original_error=None):
        message = f"Data processing failed during: {operation}"
        if original_error:
            message += f"\nOriginal error: {original_error}"
        super().__init__(message)
