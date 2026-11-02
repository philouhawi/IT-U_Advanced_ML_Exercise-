import kagglehub
import os
from MLP import mlp as run_price_prediction
from airline_classification import run as run_airline_classification
# Download latest version

FILE_NAME = 'Clean_Dataset.csv'

if __name__ == "__main__":
    path = kagglehub.dataset_download("shubhambathwal/flight-price-prediction", output_dir='data')
    src_path = os.path.join(str(os.path.abspath(os.path.curdir)), path, FILE_NAME)

    price_model = run_price_prediction(src_path)
    airline_model = run_airline_classification(src_path)