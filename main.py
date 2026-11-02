import kagglehub
import os
from MLP import MLP
from airline_classification import run as run_airline_classification
# Download latest version

FILE_NAME = 'Clean_Dataset.csv'

if __name__ == "__main__":
    path = kagglehub.dataset_download("shubhambathwal/flight-price-prediction", output_dir='data')
    mlp = MLP(os.path.join(str(os.path.abspath(os.path.curdir)), path, FILE_NAME))
    mlp.inspect_dataset()
    mlp.plot_data("price")

    mlp.data_preprocessing()
    [mlp.x_train, mlp.y_train], [mlp.x_val, mlp.y_val], [mlp.x_test, mlp.y_test] = mlp.split_data()
    mlp.linear_regression_baseline()
    model = mlp.flight_fair_mlp_model()
    mlp.train_mlp(model)

    src_path = os.path.join(str(os.path.abspath(os.path.curdir)), path, FILE_NAME)
    
    airline_model = run_airline_classification(src_path)
