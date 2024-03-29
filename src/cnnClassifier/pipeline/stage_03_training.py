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



def run_pipeline(stage_name, pipeline_instance):
    """
    Run a specific pipeline.

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
    run_pipeline("Training model", ModelTrainingPipeline())