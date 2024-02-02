from cnnClassifier.entity.config_entity import TrainingConfig
import tensorflow as tf
from pathlib import Path



class Training:
    """
    Class for managing training procedures.

    Args:
        config (TrainingConfig): Configuration object for training.

    Methods:
        get_base_model: Load the base model for training.
        train_valid_generator: Set up training and validation data generators.
        save_model: Save the trained model to a specified path.
        train: Perform the training process using the configured parameters and callbacks.
    """
    def __init__(self, config: TrainingConfig):
        """
        Initializes the Training object.

        Args:
            config (TrainingConfig): Configuration object for training.
        """
        self.config = config
    
    def get_base_model(self):
        """
        Load the base model for training.
        """
        self.model = tf.keras.models.load_model(
            self.config.updated_base_model_path
        )
    
    def train_valid_generator(self):
        """
        Set up training and validation data generators.
        """
        datagenerator_kwargs = dict(
            rescale = 1./255,
            validation_split=0.20
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

        if self.config.params_is_augmentation:
            train_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
                rotation_range=40,
                horizontal_flip=True,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
                **datagenerator_kwargs
            )
        else:
            train_datagenerator = valid_datagenerator

        self.train_generator = train_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset="training",
            shuffle=True,
            **dataflow_kwargs
        )

    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        """
        Save the trained model to a specified path.

        Args:
            path (Path): The path where the model will be saved.
            model (tf.keras.Model): The trained model to be saved.
        """
        model.save(path)


    def train(self, callback_list: list):
        """
        Perform the training process using the configured parameters and callbacks.

        Args:
            callback_list (list): List of callbacks to be applied during training.
        """
        self.steps_per_epoch = self.train_generator.samples // self.train_generator.batch_size
        self.validation_steps = self.valid_generator.samples // self.valid_generator.batch_size

        self.model.fit(
            self.train_generator,
            epochs=self.config.params_epochs,
            steps_per_epoch=self.steps_per_epoch,
            validation_steps=self.validation_steps,
            validation_data=self.valid_generator,
            callbacks=callback_list
        )

        self.save_model(
            path=self.config.trained_model_path,
            model=self.model
        )