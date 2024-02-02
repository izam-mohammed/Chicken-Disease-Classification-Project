import os
import urllib.request as request
import zipfile
from cnnClassifier import logger
from cnnClassifier.utils.common import get_size
from cnnClassifier.entity.config_entity import DataIngestionConfig
from pathlib import Path


class DataIngestion:
    """
    A class for handling data ingestion tasks, such as downloading and extracting files.

    Attributes:
        config (DataIngestionConfig): Configuration object containing data ingestion settings.
    """

    def __init__(self, config: DataIngestionConfig):
        """
        Initializes the DataIngestion object.

        Args:
            config (DataIngestionConfig): Configuration object containing data ingestion settings.
        """
        self.config = config
    
    def download_file(self) -> None:
        """
        Downloads the data file from the specified source URL.

        Returns:
            None
        """
        if not os.path.exists(self.config.local_data_file):
            filename, headers = request.urlretrieve(
                url = self.config.source_URL,
                filename = self.config.local_data_file
            )
            logger.info(f"{filename} download! with following info: \n{headers}")
        else:
            logger.info(f"File already exists of size: {get_size(Path(self.config.local_data_file))}")  

    
    def extract_zip_file(self) -> None:
        """
        Extracts the contents of the downloaded zip file.

        Returns:
            None
        """
        unzip_path = self.config.unzip_dir
        os.makedirs(unzip_path, exist_ok=True)
        with zipfile.ZipFile(self.config.local_data_file, 'r') as zip_ref:
            zip_ref.extractall(unzip_path)
            