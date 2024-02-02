import os
import urllib.request as request
from zipfile import ZipFile
import tensorflow as tf
from pathlib import Path

from cnnClassifier.entity.config_entity import PrepareBaseModelConfig


class PrepareBaseModel:
    """
    A class for preparing and updating a base model.

    Attributes:
        config (PrepareBaseModelConfig): Configuration object containing model preparation settings.

    Methods:
        __init__: Initializes the PrepareBaseModel object.
        get_base_model: Builds and saves the base model.
        _prepare_full_model: Creates and returns the full model.
        update_base_model: Upgrades from the base model to the full model.
        save_model: Saves the given model to the specified path.

    Usage:
        prepare_base_model_config = PrepareBaseModelConfig(...)  # Initialize with appropriate configuration
        prepare_base_model = PrepareBaseModel(config=prepare_base_model_config)
        prepare_base_model.get_base_model()
        prepare_base_model.update_base_model()
    """
    def __init__(self, config: PrepareBaseModelConfig):
        """
        Initializes the PrepareBaseModel object.

        Args:
            config (PrepareBaseModelConfig): Configuration object containing model preparation settings.
        """
        self.config = config
        
    def get_base_model(self):
        """
        Builds and saves the base model.

        Returns:
            None
        """
        self.model = tf.keras.applications.vgg16.VGG16(
            input_shape = self.config.params_image_size,
            weights = self.config.params_weights,
            include_top = self.config.params_include_top
        )
        
        self.save_model(path=self.config.base_model_path, model=self.model)
        
    @staticmethod
    def _prepare_full_model(model, classes, freeze_all, freeze_till, learning_rate):
        """
        Creates and returns the full model.

        Args:
            model (tf.keras.Model): Base model.
            classes (int): Number of output classes.
            freeze_all (bool): If True, freeze all layers in the model.
            freeze_till (int): If specified and greater than 0, freeze layers until this index.
            learning_rate (float): Learning rate for model compilation.

        Returns:
            tf.keras.Model: Full model.
        """
        if freeze_all:
            for layer in model.layers:
                model.trainable = False
        elif (freeze_till is not None) and (freeze_till > 0):
            for layer in model.layers[:-freeze_till]:
                model.trainable = False
        
        flatten_in = tf.keras.layers.Flatten()(model.output)
        prediction = tf.keras.layers.Dense(
            units = classes,
            activation='softmax',
        )(flatten_in)
        
        full_model = tf.keras.models.Model(
            inputs = model.input,
            outputs = prediction,
        )
        
        full_model.compile(
            optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate),
            loss = tf.keras.losses.CategoricalCrossentropy(),
            metrics=['accuracy'],
        )
        
        full_model.summary()
        return full_model
        
    def update_base_model(self):
        """
        Upgrades from the base model to the full model.

        Returns:
            None
        """
        self.full_model = self._prepare_full_model(
            model=self.model,
            classes=self.config.params_classes,
            freeze_all=True,
            freeze_till=None,
            learning_rate=self.config.params_learning_rate
        )
        self.save_model(path=self.config.updated_base_model_path, model=self.full_model)
    
    @staticmethod
    def save_model(path:Path, model:tf.keras.Model):
        """
        Saves the given model to the specified path.

        Args:
            path (Path): File path to save the model.
            model (tf.keras.Model): Model to be saved.

        Returns:
            None
        """
        model.save(path)