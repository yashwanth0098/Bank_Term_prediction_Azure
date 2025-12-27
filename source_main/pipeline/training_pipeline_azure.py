import os
import sys
from dotenv import load_dotenv

load_dotenv()

from source_main.exception.exception import BankException
from source_main.logging.logging import logging

from source_main.components.data_ingestion import DataIngestion
from source_main.components.data_validation import DataValidation
from source_main.components.data_transformation import DataTransformation
from source_main.components.data_model_trainer import ModelTrainer

from source_main.entity.config import (
    TrainingPipelineConfig,
    DataIngestionConfig,
    DataTransformationConfig,
    DataValidationConfig,
    ModelTrainerConfig
)

from source_main.entity.artifact import (
    DataIngestionArtifact,
    DataValidationArtifact,
    DataTransformationArtifact,
    ModelTrainerArtifact
)

from cloud.azure_blob_syncer import AzureBlobSync

import sys
sys.path.append('/path/to/azure/storage/blob')

class TrainingPipeline:
    def __init__(self):
        self.training_pipeline_config = TrainingPipelineConfig()
        self.blob_sync = AzureBlobSync()

    def start_data_ingestion(self) -> DataIngestionArtifact:
        try:
            logging.info("Starting data ingestion")
            config = DataIngestionConfig(
                training_pipeline_config=self.training_pipeline_config
            )
            ingestion = DataIngestion(data_ingestion_config=config)
            return ingestion.initiate_data_ingestion()
        except Exception as e:
            raise BankException(e, sys)

    def start_data_validation(
        self,
        data_ingestion_artifact: DataIngestionArtifact
    ) -> DataValidationArtifact:
        try:
            logging.info("Starting data validation")
            config = DataValidationConfig(
                training_pipeline_config=self.training_pipeline_config
            )
            validation = DataValidation(
                data_validation_config=config,
                data_ingestion_artifact=data_ingestion_artifact
            )
            return validation.initiate_data_validation()
        except Exception as e:
            raise BankException(e, sys)

    def start_data_transformation(
        self,
        data_validation_artifact: DataValidationArtifact
    ) -> DataTransformationArtifact:
        try:
            logging.info("Starting data transformation")
            config = DataTransformationConfig(
                training_pipeline_config=self.training_pipeline_config
            )
            transformation = DataTransformation(
                data_transformation_config=config,
                data_validation_artifact=data_validation_artifact
            )
            return transformation.initiate_data_transformation()
        except Exception as e:
            raise BankException(e, sys)

    def start_model_trainer(
        self,
        data_transformation_artifact: DataTransformationArtifact
    ) -> ModelTrainerArtifact:
        try:
            logging.info("Starting model training")
            config = ModelTrainerConfig(
                training_pipeline_config=self.training_pipeline_config
            )
            trainer = ModelTrainer(
                model_trainer_config=config,
                data_transformation_artifact=data_transformation_artifact
            )
            return trainer.initiate_model_trainer()
        except Exception as e:
            raise BankException(e, sys)

    def sync_artifacts_to_blob(self) -> None:
        try:
            blob_prefix = f"artifacts/{self.training_pipeline_config.timestamp}"

            self.blob_sync.sync_folder_to_blob(
                local_folder=self.training_pipeline_config.artifact_dir,
                blob_prefix=blob_prefix
            )

            logging.info("Artifacts synced to Azure Blob Storage")
        except Exception as e:
            raise BankException(e, sys)

    def sync_saved_model_dir_to_blob(self) -> None:
        try:
            blob_prefix = f"final_model/{self.training_pipeline_config.timestamp}"

            self.blob_sync.sync_folder_to_blob(
                local_folder=self.training_pipeline_config.model_dir,
                blob_prefix=blob_prefix
            )

            logging.info("Model directory synced to Azure Blob Storage")
        except Exception as e:
            raise BankException(e, sys)

    def run_pipeline(self) -> ModelTrainerArtifact:
        try:
            ingestion_artifact = self.start_data_ingestion()

            validation_artifact = self.start_data_validation(
                data_ingestion_artifact=ingestion_artifact
            )

            transformation_artifact = self.start_data_transformation(
                data_validation_artifact=validation_artifact
            )

            model_trainer_artifact = self.start_model_trainer(
                data_transformation_artifact=transformation_artifact
            )

            self.sync_artifacts_to_blob()
            self.sync_saved_model_dir_to_blob()

            return model_trainer_artifact

        except Exception as e:
            raise BankException(e, sys)
