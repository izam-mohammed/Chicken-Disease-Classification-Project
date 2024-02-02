from cnnClassifier.components.prepare_base_model import PrepareBaseModel
from cnnClassifier.config.configuration import ConfigurationManager
from cnnClassifier import logger


class PrepareBaseModelTrainingPipeline:
    """
    A training pipeline for preparing the base model.

    Methods:
        __init__: Initializes the PrepareBaseModelTrainingPipeline object.
        main: Executes the main steps for preparing the base model, including obtaining and updating the model.
    """
    def __init__(self) -> None:
        """
        Initializes the PrepareBaseModelTrainingPipeline object.
        """
        pass

    def main(self) -> None:
        """
        Executes the main steps for preparing the base model, including obtaining and updating the model.

        Returns:
            None
        """
        config = ConfigurationManager()
        prepare_base_model_config = config.get_prepare_base_model_config()
        prepare_base_model = PrepareBaseModel(config=prepare_base_model_config)
        prepare_base_model.get_base_model()
        prepare_base_model.update_base_model()