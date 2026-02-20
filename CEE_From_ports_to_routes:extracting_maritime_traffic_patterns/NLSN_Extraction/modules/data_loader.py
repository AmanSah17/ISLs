import pandas as pd
import logging
import os

class AISDataLoader:
    """
    Handles loading of AIS data from files.
    """
    def __init__(self, file_path):
        """
        Initialize the loader with the path to the data file.
        :param file_path: Absolute or relative path to the parquet file.
        """
        self.file_path = file_path
        self.logger = logging.getLogger(__name__)

    def load_data(self):
        """
        Loads data from parquet file.
        :return: DataFrame containing AIS data, or None if loading fails.
        """
        if not os.path.exists(self.file_path):
            self.logger.error(f"File not found: {self.file_path}")
            raise FileNotFoundError(f"File not found: {self.file_path}")

        self.logger.info(f"Loading data from {self.file_path}...")
        try:
            df = pd.read_parquet(self.file_path)
            self.logger.info(f"Successfully loaded {len(df)} rows.")
            
            # Basic validation
            required_cols = ['LAT', 'LON', 'MMSI']
            missing = [col for col in required_cols if col not in df.columns]
            if missing:
                self.logger.warning(f"Missing recommended columns: {missing}")
                
            return df
        except Exception as e:
            self.logger.error(f"Error loading data: {e}")
            raise e
