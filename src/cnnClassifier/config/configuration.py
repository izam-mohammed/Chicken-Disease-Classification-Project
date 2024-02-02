from cnnClassifier.constants import *
from cnnClassifier.utils.common import read_yaml, create_directories
import os
from cnnClassifier.entity.config_entity import (
    DataIngestionConfig,
    PrepareBaseModelConfig,
    PrepareCallbacksConfig,
    TrainingConfig,
    EvaluationConfig,
)


class ConfigurationManager:
    """
    A class for managing configuration settings related to the components.

    Attributes:
        config_filepath (str): File path for the main configuration file.
        params_filepath (str): File path for the parameters file.

    Methods:
        __init__: Initializes the ConfigurationManager object.
        get_data_ingestion_config: Returns data in the format of DataIngestionConfig from the config.yaml file.
        get_prepare_base_model_config: Returns data as PrepareBaseModelConfig.
        get_prepare_callback_config: Returns a PrepareCallbackConfig data type of the configuration of callbacks.
        get_training_config: Returns the configuration for training the model.
        get_validation_config: Returns an evaluation config data object.
    """
    def __init__(
        self, config_filepath=CONFIG_FILE_PATH, params_filepath=PARAMS_FILE_PATH
    ) -> None:
        """
        Initializes the ConfigurationManager object.

        Args:
            config_filepath (str, optional): File path for the main configuration file. Defaults to CONFIG_FILE_PATH.
            params_filepath (str, optional): File path for the parameters file. Defaults to PARAMS_FILE_PATH.
        """
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        create_directories([self.config.artifacts_root])

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        """
        Retrieves the data ingestion configuration.

        Returns:
            DataIngestionConfig: Object containing data ingestion configuration settings.
        """
        config = self.config.data_ingestion

        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            source_URL=config.source_URL,
            local_data_file=config.local_data_file,
            unzip_dir=config.unzip_dir,
        )

        return data_ingestion_config

    def get_prepare_base_model_config(self) -> PrepareBaseModelConfig:
        """
        Retrieves the configuration for preparing a base model.

        Returns:
            PrepareBaseModelConfig: Object containing the configuration for preparing a base model.

        """
        config = self.config.prepare_base_model

        create_directories([config.root_dir])

        prepare_base_model_config = PrepareBaseModelConfig(
            root_dir=Path(config.root_dir),
            base_model_path=Path(config.base_model_path),
            updated_base_model_path=Path(config.updated_base_model_path),
            params_image_size=self.params.IMAGE_SIZE,
            params_learning_rate=self.params.LEARNING_RATE,
            params_include_top=self.params.INCLUDE_TOP,
            params_weights=self.params.WEIGHTS,
            params_classes=self.params.CLASSES,
        )

        return prepare_base_model_config

    def get_prepare_callback_config(self) -> PrepareCallbacksConfig:
        """
        Retrieves the configuration for preparing callbacks.

        Returns:
            PrepareCallbacksConfig: Object containing the configuration for callbacks.
        """
        config = self.config.prepare_callbacks
        model_ckpt_dir = os.path.dirname(config.checkpoint_model_filepath)
        create_directories(
            [Path(model_ckpt_dir), Path(config.tensorboard_root_log_dir)]
        )

        prepare_callback_config = PrepareCallbacksConfig(
            root_dir=Path(config.root_dir),
            tensorboard_root_log_dir=Path(config.tensorboard_root_log_dir),
            checkpoint_model_filepath=Path(config.checkpoint_model_filepath),
        )

        return prepare_callback_config

    def get_training_config(self) -> TrainingConfig:
        """
        Retrieves the configuration for training.

        Returns:
            TrainingConfig: Object containing the configuration for training.
        """
        training = self.config.training
        prepare_base_model = self.config.prepare_base_model
        params = self.params
        training_data = os.path.join(
            self.config.data_ingestion.unzip_dir, "Chicken-fecal-images"
        )
        create_directories([Path(training.root_dir)])

        training_config = TrainingConfig(
            root_dir=Path(training.root_dir),
            trained_model_path=Path(training.trained_model_path),
            updated_base_model_path=Path(prepare_base_model.updated_base_model_path),
            training_data=Path(training_data),
            params_epochs=params.EPOCHS,
            params_batch_size=params.BATCH_SIZE,
            params_is_augmentation=params.AUGMENTATION,
            params_image_size=params.IMAGE_SIZE,
        )

        return training_config

    def get_validation_config(self) -> EvaluationConfig:
        """
        Returns an evaluation config data object.

        Returns:
            EvaluationConfig: Object containing the configuration for model evaluation.
        """
        eval_config = EvaluationConfig(
            path_of_model="artifacts/training/model.h5",
            training_data="artifacts/data_ingestion/Chicken-fecal-images",
            all_params=self.params,
            params_image_size=self.params.IMAGE_SIZE,
            params_batch_size=self.params.BATCH_SIZE,
        )
        return eval_config
