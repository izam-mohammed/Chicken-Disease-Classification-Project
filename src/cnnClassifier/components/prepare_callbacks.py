import os
import urllib.request as request
from zipfile import ZipFile
import tensorflow as tf
import time

from cnnClassifier.entity.config_entity import PrepareCallbacksConfig



class PrepareCallback:
    """
    A class for preparing callbacks during model training.

    Attributes:
        config (PrepareCallbacksConfig): Configuration object containing callback settings.

    Methods:
        __init__: Initializes the PrepareCallback object.
        _create_tb_callbacks: Creates and returns TensorBoard callback.
        _create_ckpt_callbacks: Creates and returns ModelCheckpoint callback.
        get_tb_ckpt_callbacks: Retrieves a list of TensorBoard and ModelCheckpoint callbacks.

    Usage:
        prepare_callback_config = PrepareCallbacksConfig(...)  # Initialize with appropriate configuration
        prepare_callback = PrepareCallback(config=prepare_callback_config)
        tb_ckpt_callbacks = prepare_callback.get_tb_ckpt_callbacks()
    """
    def __init__(self, config: PrepareCallbacksConfig) -> None:
        """
        Initializes the PrepareCallback object.

        Args:
            config (PrepareCallbacksConfig): Configuration object containing callback settings.
        """
        self.config = config


    
    @property
    def _create_tb_callbacks(self):
        """
        Creates and returns TensorBoard callback.

        Returns:
            tf.keras.callbacks.TensorBoard: TensorBoard callback.
        """
        timestamp = time.strftime("%Y-%m-%d-%H-%M-%S")
        tb_running_log_dir = os.path.join(
            self.config.tensorboard_root_log_dir,
            f"tb_logs_at_{timestamp}",
        )
        return tf.keras.callbacks.TensorBoard(log_dir=tb_running_log_dir)
    

    @property
    def _create_ckpt_callbacks(self):
        """
        Creates and returns ModelCheckpoint callback.

        Returns:
            tf.keras.callbacks.ModelCheckpoint: ModelCheckpoint callback.
        """
        return tf.keras.callbacks.ModelCheckpoint(
            filepath=str(self.config.checkpoint_model_filepath),
            save_best_only=True
        )


    def get_tb_ckpt_callbacks(self):
        """
        Retrieves a list of TensorBoard and ModelCheckpoint callbacks.

        Returns:
            list: List containing TensorBoard and ModelCheckpoint callbacks.

        Usage:
        tb_ckpt_callbacks = prepare_callback.get_tb_ckpt_callbacks()
        """
        return [
            self._create_tb_callbacks,
            self._create_ckpt_callbacks
        ]
