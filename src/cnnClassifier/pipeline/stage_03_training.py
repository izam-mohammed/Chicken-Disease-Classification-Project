from cnnClassifier.config.configuration import ConfigurationManager
from cnnClassifier.components.prepare_callbacks import PrepareCallback
from cnnClassifier.components.training import Training
from cnnClassifier import logger

class ModelTrainingPipeline:
    """
    A training pipeline for the model.

    Methods:
        __init__: Initializes the ModelTrainingPipeline object.
        main: Executes the main steps for model training, including preparing callbacks, obtaining the base model,
              and training the model with generators.
    """
    def __init__(self) -> None:
        """
        Initializes the ModelTrainingPipeline object.
        """
        pass

    def main(self) -> None:
        """
        Executes the main steps for model training, including preparing callbacks, obtaining the base model,
        and training the model with generators.

        Return:
            None
        """
        config = ConfigurationManager()
        prepare_callbacks_config = config.get_prepare_callback_config()
        prepare_callbacks = PrepareCallback(config=prepare_callbacks_config)
        callback_list = prepare_callbacks.get_tb_ckpt_callbacks()

        training_config = config.get_training_config()
        training = Training(config=training_config)
        training.get_base_model()
        training.train_valid_generator()
        training.train(callback_list=callback_list)
