import tensorflow as tf
from pathlib import Path
from cnnClassifier.entity.config_entity import EvaluationConfig
from cnnClassifier.utils.common import save_json


# component
class Evaluation:
    """
    A class for evaluating a trained model.

    Methods:
        __init__: Initializes the Evaluation object.
        _valid_generator: Generates data for validation and stores it in the valid generator.
        load_model: Loads a saved Keras model.
        evaluation: Evaluates the loaded model on the validation data and saves the scores.
        save_score: Saves the evaluation scores as a JSON file.

    Attributes:
        config (EvaluationConfig): Object containing the configuration for model evaluation.
        valid_generator: Data generator for validation data.
        model (tf.keras.Model): Loaded Keras model.
        score (list): Evaluation scores for loss and accuracy.

    """
    def __init__(self, config: EvaluationConfig):
        """
        Initializes the Evaluation object.

        Args:
            config (EvaluationConfig): Object containing the configuration for model evaluation.
        """
        self.config = config

    
    def _valid_generator(self):
        """
        Generate data for validation and store it to valid generator
        """
        datagenerator_kwargs = dict(
            rescale = 1./255,
            validation_split=0.30
        )

        dataflow_kwargs = dict(
            target_size=self.config.params_image_size[:-1],
            batch_size=self.config.params_batch_size,
            interpolation="bilinear"
        )

        valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
            **datagenerator_kwargs
        )

        self.valid_generator = valid_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset="validation",
            shuffle=False,
            **dataflow_kwargs
        )

    
    @staticmethod
    def load_model(path: Path) -> tf.keras.Model:
        """
        Loads a pre-trained model.

        Args:
            path (Path): Path to the saved model file.

        Returns:
            tf.keras.Model: Loaded pre-trained model.
        """
        return tf.keras.models.load_model(path)
    

    def evaluation(self):
        """ 
        Evaluating the model and save it to score
        """ 
        self.model = self.load_model(self.config.path_of_model)
        self._valid_generator()
        self.score = self.model.evaluate(self.valid_generator)

    
    def save_score(self):
        """ 
        Saving score as a json file
        """
        scores = {"loss": self.score[0], "accuracy": self.score[1]}
        save_json(path=Path("scores.json"), data=scores)