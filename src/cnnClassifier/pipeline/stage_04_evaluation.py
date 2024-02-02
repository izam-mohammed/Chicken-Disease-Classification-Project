from cnnClassifier.config.configuration import ConfigurationManager
from cnnClassifier.components.evaluation import Evaluation
from cnnClassifier import logger


class EvaluationPipeline:
    """
    An evaluation pipeline for the model.

    Methods:
        __init__: Initializes the EvaluationPipeline object.
        main: Executes the main steps for model evaluation, including obtaining the validation configuration,
              performing evaluation, and saving the evaluation score.
    """
    def __init__(self) -> None:
        """
        Initializes the EvaluationPipeline object.
        """
        pass

    def main(self) -> None:
        """
        Executes the main steps for model evaluation, including obtaining the validation configuration,
        performing evaluation, and saving the evaluation score.

        Returns:
            None
        """
        config = ConfigurationManager()
        val_config = config.get_validation_config()
        evaluation = Evaluation(val_config)
        evaluation.evaluation()
        evaluation.save_score()


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
    run_pipeline("Evaluating model", EvaluationPipeline())