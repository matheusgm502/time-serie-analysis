import numpy as np

def calculate_metrics(actual, predicted):
    # Mean Absolute Error, ou erro absoluto médio
    mae = np.mean(np.abs(actual - predicted[2:]))

    # Mean Absolute Percentage Error, ou erro absoluto percentual médio
    mape = np.mean(np.abs((actual - predicted[2:]) / actual))

    # Mean Squared Error, ou erro quadrático médio
    mse = np.mean((actual - predicted[2:])**2)

    # Root Mean Squared Error, ou raiz do erro quadrático médio
    rmse = np.sqrt(mse)

    return mae, mape, mse, rmse