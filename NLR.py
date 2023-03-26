import os
from random import random

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

    def residual_boosting(self, iterations=5):
        boosting_results = []

        for date in self.date_list:
            x = self.X.loc[date, :].to_numpy()
            v = self.V.loc[date, :].to_numpy()
            s = self.S.loc[date, :].to_numpy()
            h = self.H.loc[date, :].to_numpy()
            x_data = np.vstack((x, s, v)).T
            y_data = h

            best_params = None
            best_residuals = None
            min_residual_sum = np.inf

            for _ in range(iterations):
                def wrapper(x, eta, beta):
                    return self.model(x[:, 0], eta, x[:, 1], x[:, 2], beta)

                p0 = [0.1, 0.1]  # Initial guess for the parameters
                params, _ = curve_fit(wrapper, x_data, y_data, p0=p0, maxfev=10000)

                y_pred = wrapper(x_data, *params)
                residuals = y_data - y_pred

                residual_sum = np.sum(np.abs(residuals))
                if residual_sum < min_residual_sum:
                    min_residual_sum = residual_sum
                    best_params = params
                    best_residuals = residuals

                # Choose a random residual and add it to y_data
                random_residual = np.random.choice(residuals, size=residuals.shape)
                y_data += random_residual

            boosting_results.append((date, best_params, best_residuals))

        return boosting_results

    def NLR(self, bootstrape=None):
        if bootstrape == 'residual':
            return self.residual_boosting(iterations=5)

        if bootstrape == 'pair':
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
            def wrapper(x, eta, beta):
                return self.model(x[:, 0], eta, x[:, 1], x[:, 2], beta)

            p0 = [0.1, 0.1]  # Initial guess for the parameters
            params, _ = curve_fit(wrapper, x_data, y_data, p0=p0, maxfev=10000)

        return params[0], params[1]


# Create and run the NonLinearRegression
nonlinear_regression = NonLinearRegression(X, V, S, H, model_function)
print(nonlinear_regression.NLR(bootstrape='residual'))
