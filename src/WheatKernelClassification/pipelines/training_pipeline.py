from src.WheatKernelClassification.components.data_ingestion import DataIngestion

import os
import sys
from src.WheatKernelClassification.logger import logging
from src.WheatKernelClassification.exception import customexception
import pandas as pd

obj = DataIngestion()

obj.initiate_data_ingestion()