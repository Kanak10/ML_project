import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig

# this class provide all the input things required for this data ingestion component
@dataclass
class DataIngestionConfig:
    train_data_split: str = os.path.join("artifacts", "train.csv")
    test_data_split: str = os.path.join("artifacts", "test.csv")
    raw_data_split: str = os.path.join("artifacts", "data.csv")

class DataIngetion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_integestion(self):
        logging.info("Enteres the data integestion method or component")
        try:
            df = pd.read_csv("notebook/data/stud.csv")
            logging.info("Read the dataset as dataframe")

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_split), exist_ok=True)
            
            df.to_csv(self.ingestion_config.raw_data_split, index=False, header=False)

            logging.info("Train test split initiated")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_split, index=False, header=True)
            
            train_set.to_csv(self.ingestion_config.test_data_split, index=False, header=True)

            logging.info("Ingestion of the data is completed")

            return (
                self.ingestion_config.train_data_split,
                self.ingestion_config.test_data_split
            )
        except Exception as e:
            raise CustomException(e, sys)
        
if __name__ == "__main__":
    obj = DataIngetion()
    train_data, test_data = obj.initiate_data_integestion()

    data_transformation = DataTransformation()
    data_transformation.initiate_data_transformation(train_data, test_data)