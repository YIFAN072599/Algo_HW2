import os
from random import random

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy import stats

from statsmodels.stats.diagnostic import het_breuschpagan

WORK_DIR = os.path.dirname(__file__)
REGRESSION_DIR = os.path.join(WORK_DIR, 'regression_data')

X = pd.read_csv(os.path.join(REGRESSION_DIR, 'arrival_price.csv'), index_col=0)
V = pd.read_csv(os.path.join(REGRESSION_DIR, 'total_volume.csv'), index_col=0)
S = pd.read_csv(os.path.join(REGRESSION_DIR, 'return_std.csv'), index_col=0)
H = pd.read_csv(os.path.join(REGRESSION_DIR, 'temporary_impact.csv'), index_col=0)


# Define your nonlinear model function
def model_function(x, eta, s, v, beta):
    return eta * s * np.sign(x) * (abs(x) / (6 / 6.5) * v) ** beta


def calculate_t_values(eta_list, beta_list):
    eta_mean = np.mean(eta_list)
    eta_std = np.std(eta_list, ddof=1)
    beta_mean = np.mean(beta_list)
    beta_std = np.std(beta_list, ddof=1)

    eta_t_value = eta_mean / (eta_std / np.sqrt(len(eta_list)))
    beta_t_value = beta_mean / (beta_std / np.sqrt(len(beta_list)))

    return eta_t_value, beta_t_value


class NonLinearRegression():
    def __init__(self, X, V, S, H, model_function):
        self.X = X
        self.V = V
        self.S = S
        self.H = H
        self.date_list = X.index
        self.model = model_function
        self.nonl_ticker, self.l_ticker = self.split_astock()

    def residual_boosting(self, X, V, S, H, parameter_bounds, iterations=50):
        bootstrap_results = []

        for date in self.date_list:
            x = X.loc[date, :].to_numpy()
            v = V.loc[date, :].to_numpy()
            s = S.loc[date, :].to_numpy()
            h = H.loc[date, :].to_numpy()
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

                p0 = [0.142, 0.6]  # Initial guess for the parameters
                params, _ = curve_fit(wrapper, x_data, y_data, p0=p0, maxfev=50000, bounds=parameter_bounds)

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

            # Residual Stat Test

            # test for heteroskedasticity
            X_with_constant = np.column_stack((np.ones(len(x_data)), x_data))
            bp_test = het_breuschpagan(residuals, X_with_constant)

            # Shapiro-Wilk test
            stat, sw_p_value = stats.shapiro(residuals)

            # Results
            bp_stat, bp_p_value, bp_f_stat, bp_f_p_value = bp_test
            eta_t_value, beta_t_value = calculate_t_values(eta_list, beta_list)

            bootstrap_results.append((date, params_mean[0], eta_t_value, params_mean[1], beta_t_value,
                                      bp_p_value, sw_p_value))

        return pd.DataFrame(bootstrap_results,
                            columns=['Date', 'eta', 'eta_t', 'beta', 'beta_t', 'heteroskedasticity_p_pvalue',
                                     'shapiro_wilk_p_value'])

    def paired_bootstrap(self, X, V, S, H, parameter_bounds, iterations=50):
        bootstrap_results = []
        for date in self.date_list:
            x = X.loc[date, :].to_numpy()
            v = V.loc[date, :].to_numpy()
            s = S.loc[date, :].to_numpy()
            h = H.loc[date, :].to_numpy()
            data = np.vstack((x, s, v, h)).T
            data = data[~np.isnan(data).any(axis=1)]

            params_sum = np.zeros(2)
            eta_list = []
            beta_list = []

            for _ in range(iterations):
                # Generate a random index for paired bootstrap
                random_index = np.random.choice(data.shape[0], size=data.shape[0], replace=True)
                bootstrap_data = data[random_index, :]
                x_data = bootstrap_data[:, :3]
                y_data = bootstrap_data[:, -1]

                def wrapper(x, eta, beta):
                    return self.model(x[:, 0], eta, x[:, 1], x[:, 2], beta)

                p0 = [0.146, 0.6]  # Initial guess for the parameters
                params, _ = curve_fit(wrapper, x_data, y_data, p0=p0, maxfev=10000, bounds=parameter_bounds)

                eta_list.append(params[0])
                beta_list.append(params[1])

                params_sum += params

            params_mean = params_sum / iterations
            x_data = data[:, :3]
            y_data = data[:, -1]
            y_pred = wrapper(x_data, *params_mean)
            residuals = y_data - y_pred

            # Residual Stat Test

            # test for heteroskedasticity
            X_with_constant = np.column_stack((np.ones(len(x_data)), x_data))
            bp_test = het_breuschpagan(residuals, X_with_constant)

            # Shapiro-Wilk test
            stat, sw_p_value = stats.shapiro(residuals)

            # Results
            bp_stat, bp_p_value, bp_f_stat, bp_f_p_value = bp_test
            eta_t_value, beta_t_value = calculate_t_values(eta_list, beta_list)

            bootstrap_results.append(
                (date, params_mean[0], eta_t_value, params_mean[1], beta_t_value,
                 bp_p_value, sw_p_value))

        return pd.DataFrame(bootstrap_results,
                            columns=['Date', 'eta', 'eta_t', 'beta', 'beta_t', 'heteroskedasticity_p_pvalue',
                                     'shapiro_wilk_p_value'])

    def NLR(self, bootstrape=None, liquid=None):
        # Set the bounds for eta and beta
        low_bounds = (0, 0)
        up_bounds = (1, 1)

        # Combine the bounds into a single tuple
        parameter_bounds = (low_bounds, up_bounds)

        if liquid == 'high':
            X = self.X[self.l_ticker]
            V = self.V[self.l_ticker]
            H = self.H[self.l_ticker]
            S = self.S[self.l_ticker]
        elif liquid == 'low':
            X = self.X[self.nonl_ticker]
            V = self.V[self.nonl_ticker]
            H = self.H[self.nonl_ticker]
            S = self.S[self.nonl_ticker]
        else:
            X = self.X
            V = self.V
            H = self.H
            S = self.S
        if bootstrape == 'residual':
            return self.residual_boosting(X, V, S, H, parameter_bounds, iterations=10)

        if bootstrape == 'pair':
            return self.paired_bootstrap(X, V, S, H, parameter_bounds, iterations=10)

        results = []
        for date in self.date_list:
            x = X.loc[date, :].to_numpy()
            v = V.loc[date, :].to_numpy()
            s = S.loc[date, :].to_numpy()
            h = H.loc[date, :].to_numpy()

            # Prepare the data for curve_fit
            data = np.vstack((x, s, v, h)).T
            data = data[~np.isnan(data).any(axis=1)]
            x_data = data[:, :3]
            y_data = data[:, -1]

            # Fit the nonlinear regression model
            def wrapper(x, eta, beta):
                return self.model(x[:, 0], eta, x[:, 1], x[:, 2], beta)

            p0 = [0.146, 0.6]  # Initial guess for the parameters
            params, _ = curve_fit(wrapper, x_data, y_data, p0=p0, maxfev=10000, bounds=parameter_bounds)

            y_pred = wrapper(x_data, *params)
            residuals = y_data - y_pred

            # Residual Stat Test

            # test for heteroskedasticity
            X_with_constant = np.column_stack((np.ones(len(x_data)), x_data))
            bp_test = het_breuschpagan(residuals, X_with_constant)

            # Shapiro-Wilk test
            stat, sw_p_value = stats.shapiro(residuals)
            # Results
            bp_stat, bp_p_value, bp_f_stat, bp_f_p_value = bp_test

            results.append(
                (date, params[0], params[1], bp_p_value, sw_p_value))

        return pd.DataFrame(results,
                            columns=['Date', 'eta', 'beta', 'heteroskedasticity_p_pvalue', 'shapiro_wilk_p_value'])

    def split_astock(self, k=250):
        less_active = list(self.V.mean().sort_values()[:k].index)
        more_active = list(self.V.mean().sort_values()[-k:].index)

        return less_active, more_active


# Create and run the NonLinearRegression
nonlinear_regression = NonLinearRegression(X, V, S, H, model_function)
print('_____________________high_________________________________')
df_high = nonlinear_regression.NLR(bootstrape='residual', liquid='high')
eta_mean = df_high['eta'].mean()
beta_mean = df_high['beta'].mean()
print(f'eta: {eta_mean}, beta: {beta_mean}')
print(df_high)
print('_____________________low_________________________________')
df_low = nonlinear_regression.NLR(bootstrape='residual', liquid='low')
eta_mean = df_low['eta'].mean()
beta_mean = df_low['beta'].mean()
print(f'eta: {eta_mean}, beta: {beta_mean}')
print(df_low)
print('_____________________all_________________________________')
df = nonlinear_regression.NLR(bootstrape='residual')
eta_mean = df['eta'].mean()
beta_mean = df['beta'].mean()
print(f'eta: {eta_mean}, beta: {beta_mean}')
print(df)
