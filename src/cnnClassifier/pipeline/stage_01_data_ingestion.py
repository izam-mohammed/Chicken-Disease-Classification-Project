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



def run_pipeline(stage_name, pipeline_instance):
    """
    Run a specific  pipeline.

    Parameters:
    - stage_name: str
        Name of the pipeline stage.
    - pipeline_instance: object
        Instance of the pipeline stage to be executed.

    Returns:
        None
    """
    try:
        logger.info(f">>>>>> Stage {stage_name} started <<<<<<")
        pipeline_instance.main()
        logger.info(f">>>>>> Stage {stage_name} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e


if __name__ == "__main__":
    run_pipeline("Data Ingestion", DataIngestionTrainingPipeline())