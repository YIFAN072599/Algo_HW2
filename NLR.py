import os

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

WORK_DIR = os.path.dirname(__file__)
REGRESSION_DIR = os.path.join(WORK_DIR, 'regression_data')

X = pd.read_csv(os.path.join(REGRESSION_DIR, 'arrival_price.csv'), index_col=0)
V = pd.read_csv(os.path.join(REGRESSION_DIR, 'total_volume.csv'), index_col=0)
S = pd.read_csv(os.path.join(REGRESSION_DIR, 'return_std.csv'), index_col=0)
H = pd.read_csv(os.path.join(REGRESSION_DIR, 'temporary_impact.csv'), index_col=0)


# Define your nonlinear model function
def model_function(x, eta, s, v, beta):
    return eta * s * (x / (6 / 6.5) * v) ** beta


class NonLinearRegression():
    def __init__(self, X, V, S, H, model_function):
        self.X = X
        self.V = V
        self.S = S
        self.H = H
        self.date_list = X.index
        self.model = model_function

    def NLR(self, boosting=None):
        if boosting == 'residual':
            return
        if boosting == 'pair':
            return

        for date in self.date_list:
            x = self.X.loc[date, :].to_numpy()
            v = self.V.loc[date, :].to_numpy()
            s = self.S.loc[date, :].to_numpy()
            h = self.H.loc[date, :].to_numpy()

            # Prepare the data for curve_fit
            x_data = np.vstack((x, s, v)).T
            y_data = h

            # Fit the nonlinear regression model
            p0 = [1.0, 1.0]  # Initial guess for the parameters
            params, _ = curve_fit(lambda x, eta, beta: self.model(x[:, 0], eta, x[:, 1], x[:, 2], beta), x_data, y_data,
                                  p0=p0)

            # Print the estimated parameters
            print(f"Date: {date}, eta: {params[0]}, beta: {params[1]}")


# Create and run the NonLinearRegression
nonlinear_regression = NonLinearRegression(X, V, S, H, model_function)
nonlinear_regression.NLR()
