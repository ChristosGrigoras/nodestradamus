"""Main module for sample docs testing."""


def process_data(input_data: dict) -> dict:
    """Process input data.

    Args:
        input_data: The data to process.

    Returns:
        Processed data dictionary.
    """
    return {"processed": True, "data": input_data}


class DataProcessor:
    """Main data processing class."""

    def __init__(self) -> None:
        """Initialize the processor."""
        self.cache: dict = {}

    def run(self, data: dict) -> dict:
        """Run the processor on data."""
        return process_data(data)
