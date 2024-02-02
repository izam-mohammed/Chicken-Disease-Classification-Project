from cnnClassifier.components.data_ingestion import DataIngestion
from cnnClassifier.config.configuration import ConfigurationManager
from cnnClassifier import logger


class DataIngestionTrainingPipeline:
    """
    A training pipeline for data ingestion.

    Methods:
        __init__: Initializes the DataIngestionTrainingPipeline object.
        main: Executes the main data ingestion steps, including downloading and extracting files.
    """
    def __init__(self) -> None:
        """
        Initializes the DataIngestionTrainingPipeline object.
        """
        pass

    def main(self) -> None:
        """
        Executes the main data ingestion steps, including downloading and extracting files.

        Return: 
            None
        """
        config = ConfigurationManager()
        data_ingestion_config = config.get_data_ingestion_config()
        data_ingestion = DataIngestion(config=data_ingestion_config)
        data_ingestion.download_file()
        data_ingestion.extract_zip_file()
