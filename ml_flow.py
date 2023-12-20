import mlflow
from mlflow.models import infer_signature
import pickle
import pandas as pd
from helper_functions import calculate_metrics

def main():
    with open('model/modelo_mlp.pkl', 'rb') as model_file:
        modelo_mlp = pickle.load(model_file)

    X_test = pd.read_csv('data/x.csv')
    y_test = pd.read_csv('data/y.csv')

    with mlflow.start_run():
        predictions = modelo_mlp.predict(X_test.values)
        mae, mape, mse, rmse = calculate_metrics(y_test.values, predictions)

        mlflow.log_metric("MAE", mae)
        mlflow.log_metric("MAPE", mape)
        mlflow.log_metric("MSE", mse)
        mlflow.log_metric("RMSE", rmse)

        # Infer the model signature (optional but recommended)
        signature = infer_signature(X_test, predictions)

        # Log the model with signature
        mlflow.sklearn.log_model(modelo_mlp, "model", signature=signature)

if __name__ == "__main__":
    main()
