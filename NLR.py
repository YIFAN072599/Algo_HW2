import os
from random import random

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import shapiro

WORK_DIR = os.path.dirname(__file__)
REGRESSION_DIR = os.path.join(WORK_DIR, 'regression_data')

X = pd.read_csv(os.path.join(REGRESSION_DIR, 'arrival_price.csv'), index_col=0)
V = pd.read_csv(os.path.join(REGRESSION_DIR, 'total_volume.csv'), index_col=0)
S = pd.read_csv(os.path.join(REGRESSION_DIR, 'return_std.csv'), index_col=0)
H = pd.read_csv(os.path.join(REGRESSION_DIR, 'temporary_impact.csv'), index_col=0)


# Define your nonlinear model function
def model_function(x, eta, s, v, beta):
    return eta * s * np.sign(x) * (abs(x) / (6 / 6.5) * v) ** beta


class NonLinearRegression():
    def __init__(self, X, V, S, H, model_function):
        self.X = X
        self.V = V
        self.S = S
        self.H = H
        self.date_list = X.index
        self.model = model_function

    def residual_boosting(self, iterations=5):
        bootstrap_results = []

        for date in self.date_list:
            x = self.X.loc[date, :].to_numpy()
            v = self.V.loc[date, :].to_numpy()
            s = self.S.loc[date, :].to_numpy()
            h = self.H.loc[date, :].to_numpy()
            data = np.vstack((x, s, v, h)).T
            data = data[~np.isnan(data).any(axis=1)]
            x_data = data[:, :3]
            y_data = data[:, -1]

            params_sum = np.zeros(2)
            eta_list = []
            beta_list = []

            for _ in range(iterations):
                def wrapper(x, eta, beta):
                    return self.model(x[:, 0], eta, x[:, 1], x[:, 2], beta)

                p0 = [0.5, 0.5]  # Initial guess for the parameters
                params, _ = curve_fit(wrapper, x_data, y_data, p0=p0, maxfev=10000)

                y_pred = wrapper(x_data, *params)
                residuals = y_data - y_pred

                # Choose a random residual and add it to y_data
                random_residual = np.random.choice(residuals, size=residuals.shape)
                y_data += random_residual

                params_sum += params
                eta_list.append(params[0])
                beta_list.append(params[1])

            params_mean = params_sum / iterations
            x_data = data[:, :3]
            y_data = data[:, -1]
            y_pred = wrapper(x_data, *params_mean)
            residuals = y_data - y_pred
            # Perform the Shapiro-Wilk test
            stat, p_value = shapiro(residuals)

            bootstrap_results.append((date, params_mean[0], params_mean[1], p_value))

            return pd.DataFrame(bootstrap_results, columns=['Date', 'eta', 'beta', 'Residual_pvalue'])

    def paired_bootstrap(self, iterations=5):
        bootstrap_results = []

        for date in self.date_list:
            x = self.X.loc[date, :].to_numpy()
            v = self.V.loc[date, :].to_numpy()
            s = self.S.loc[date, :].to_numpy()
            h = self.H.loc[date, :].to_numpy()
            data = np.vstack((x, s, v, h)).T
            data = data[~np.isnan(data).any(axis=1)]

            params_sum = np.zeros(2)

            for _ in range(iterations):
                # Generate a random index for paired bootstrap
                random_index = np.random.choice(data.shape[0], size=data.shape[0], replace=True)
                bootstrap_data = data[random_index, :]
                x_data = bootstrap_data[:, :3]
                y_data = bootstrap_data[:, -1]

                def wrapper(x, eta, beta):
                    return self.model(x[:, 0], eta, x[:, 1], x[:, 2], beta)

                p0 = [0.05, 0.5]  # Initial guess for the parameters
                params, _ = curve_fit(wrapper, x_data, y_data, p0=p0, maxfev=10000)

                params_sum += params

            params_mean = params_sum / iterations
            x_data = data[:, :3]
            y_data = data[:, -1]
            y_pred = wrapper(x_data, *params_mean)
            residuals = y_data - y_pred

            # Perform the Shapiro-Wilk test
            stat, p_value = shapiro(residuals)

            bootstrap_results.append((date, params_mean[0], params_mean[1], p_value))

        return pd.DataFrame(bootstrap_results, columns=['Date', 'eta', 'beta', 'Residual_pvalue'])

    def NLR(self, bootstrape=None):
        if bootstrape == 'residual':
            return self.residual_boosting(iterations=5)

        if bootstrape == 'pair':
            return self.paired_bootstrap(iterations=5)

        for date in self.date_list:
            x = self.X.loc[date, :].to_numpy()
            v = self.V.loc[date, :].to_numpy()
            s = self.S.loc[date, :].to_numpy()
            h = self.H.loc[date, :].to_numpy()

            # Prepare the data for curve_fit
            data = np.vstack((x, s, v, h)).T
            data = data[~np.isnan(data).any(axis=1)]
            x_data = data[:, :3]
            y_data = data[:, -1]

            # Fit the nonlinear regression model
            def wrapper(x, eta, beta):
                return self.model(x[:, 0], eta, x[:, 1], x[:, 2], beta)

            p0 = [0.1, 0.1]  # Initial guess for the parameters
            params, _ = curve_fit(wrapper, x_data, y_data, p0=p0, maxfev=10000)

        return params[0], params[1]


# Create and run the NonLinearRegression
nonlinear_regression = NonLinearRegression(X, V, S, H, model_function)
print(nonlinear_regression.NLR(bootstrape='pair'))
