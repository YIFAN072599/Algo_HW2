import pandas as pd
import numpy as np
import os
from scipy.optimize import curve_fit, least_squares

WORK_DIR = os.path.dirname(__file__)
REGRESSION_DIR = os.path.join(WORK_DIR, 'regression_data')
class TAQRegression(object):
    def __init__(self):
        self.arrival_price = pd.read_csv(os.path.join(REGRESSION_DIR,"arrival_price.csv"),index_col = 0)
        self.market_imbalance = pd.read_csv(os.path.join(REGRESSION_DIR,"market_imbalance.csv"),index_col = 0)
        self.return_std = pd.read_csv(os.path.join(REGRESSION_DIR,"return_std.csv"),index_col = 0)
        self.vwap330 = pd.read_csv(os.path.join(REGRESSION_DIR,"vwap330.csv"),index_col = 0)
        self.vwap400 = pd.read_csv(os.path.join(REGRESSION_DIR,"vwap400.csv"),index_col = 0)
        self.temporary_impact = pd.read_csv(os.path.join(REGRESSION_DIR,"temporary_impact.csv"),index_col = 0)
        self.total_volume = pd.read_csv(os.path.join(REGRESSION_DIR,"total_volume.csv"),index_col = 0)
        self.terminal_price = pd.read_csv(os.path.join(REGRESSION_DIR,"terminal_price.csv"),index_col = 0)

    def preprocessing(self, ticker_list=None, splitting_by_liquidity=False):

        X = self.vwap400 * self.market_imbalance
        print(X)
        h = self.temporary_impact
        s = self.return_std
        V = self.vwap400 * self.total_volume
        print(V)
        def clean_matrix(X, h, s, V, max_null_in_day=40, max_null_in_ticker=2):
            # we want to drop the ticker column with too much null values
            col_nulls_X = X.isnull().sum(axis=0)
            col_nulls_h = h.isnull().sum(axis=0)
            col_nulls_s = s.isnull().sum(axis=0)
            col_nulls_V = V.isnull().sum(axis=0)
            temp = pd.concat(
                [col_nulls_X[col_nulls_X >= max_null_in_ticker], col_nulls_h[col_nulls_h >= max_null_in_ticker],
                 col_nulls_s[col_nulls_s >= max_null_in_ticker], col_nulls_V[col_nulls_V >= max_null_in_ticker]])
            drop_cols = list(temp.index.unique())
            X.drop(columns=drop_cols, inplace=True)
            h.drop(columns=drop_cols, inplace=True)
            s.drop(columns=drop_cols, inplace=True)
            V.drop(columns=drop_cols, inplace=True)

            # we want to drop the day row with too much null values
            row_nulls_X = X.isnull().sum(axis=1)
            row_nulls_h = h.isnull().sum(axis=1)
            row_nulls_s = s.isnull().sum(axis=1)
            row_nulls_V = V.isnull().sum(axis=1)
            temp = pd.concat([row_nulls_X[row_nulls_X >= max_null_in_day], row_nulls_h[row_nulls_h >= max_null_in_day],
                              row_nulls_s[row_nulls_s >= max_null_in_day], row_nulls_V[row_nulls_V >= max_null_in_day]])
            drop_rows = list(temp.index.unique())
            X.drop(drop_rows, inplace=True)
            h.drop(drop_rows, inplace=True)
            s.drop(drop_rows, inplace=True)
            V.drop(drop_rows, inplace=True)

            # fill the rest Nans with daily average
            def fillna(x):
                z = x.fillna(x.mean())
                return z

            X = X.apply(lambda x: fillna(x), axis=1)
            h = h.apply(lambda x: fillna(x), axis=1)
            s = s.apply(lambda x: fillna(x), axis=1)
            V = V.apply(lambda x: fillna(x), axis=1)

            return X, h, s, V


        X_, h_, s_, V_ = clean_matrix(X, h, s, V)

        if splitting_by_liquidity:
            X_ = X_[X_.columns[X_.columns.isin(ticker_list)]]
            h_ = h_[h_.columns[h_.columns.isin(ticker_list)]]
            s_ = s_[s_.columns[s_.columns.isin(ticker_list)]]
            V_ = V_[V_.columns[V_.columns.isin(ticker_list)]]


        return X_, h_, s_, V_

    def NLR(self):
        X_, h_, s_, V_ = self.preprocessing()
        self.optimization(X_, h_, s_, V_)
        return

    def optimization(self, X_, h_, s_, V_):
        def curve_fit_func(X, eta, beta):
            X_, s_, V_ = X
            return eta * s_ * np.sign(X_) * (np.abs(X_) / ((6 / 6.5) * V_)) ** beta

        InitialParameter = (0.5, 0.5)

        model_df = pd.DataFrame({"imbalance_value": X_,
                                 "temp_impact": h_,
                                 "daily_std": s_,
                                 "daily_volume": V_})

        least_square_func = lambda p, x, y: y - p[0] * x["daily_std"] * np.sign(x["imbalance_value"]) * \
                                            (np.abs(x["imbalance_value"]) / ((6 / 6.5) * x["daily_volume"])) ** p[1]

        # curve fit the test data
        #print(InitialParameter)
        fittedParameters, pcov = curve_fit(curve_fit_func, (X_, s_, V_), h_, InitialParameter, bounds=(-1, 1.5))

        # We also implement the scipy least_square optimization method to validate our result from curve_fit
        # feel free to comment out lines of code and check the result
        # parameters = least_squares(least_square_func, InitialParameter, args=(model_df, model_df["temp_impact"]))

        print('optimized parameters through scipy curve_fit is {}'.format(fittedParameters))
        # print("optimized parameters through scipy least_squares is {}".format(parameters.x))

        modelPredictions = curve_fit_func((X_, s_, V_), *fittedParameters)

        Error = modelPredictions - h_

        SE = np.square(Error)  # squared errors
        MSE = np.mean(SE)  # mean squared errors
        RMSE = np.sqrt(MSE)  # Root Mean Squared Error, RMSE
        Rsquared = 1.0 - (np.var(Error) / np.var(h_))
        # Rsquared = 1.0 - (np.sum(SE) / np.var(h_))
        print('RMSE:', RMSE)
        print('R-squared:', Rsquared)
        print()

        return fittedParameters, Error


    def bootstrapping_residual(self, m):

        X_, h_, s_, V_ = self.preprocessing()

        def curve_fit_func(X, eta, beta):
            X_, s_, V_ = X
            return eta * s_ * np.sign(X_) * (np.abs(X_) / ((6 / 6.5) * V_)) ** beta

        fittedParameters, Error = self.optimization(X_, h_, s_, V_)

        eta_list = []
        beta_list = []
        for i in range(m):
            random_errors = np.random.choice(Error, len(Error))
            random_h_ = curve_fit_func((X_, s_, V_), *fittedParameters) + random_errors
            new_params, error = self.optimization(X_, random_h_, s_, V_)
            eta_list.append(new_params[0])
            beta_list.append(new_params[1])

        return eta_list, beta_list

    def bootstrapping_paired(self, m):

        X_, h_, s_, V_ = self.preprocessing()

        def curve_fit_func(X, eta, beta):
            X_, s_, V_ = X
            return eta * s_ * np.sign(X_) * (np.abs(X_) / ((6 / 6.5) * V_)) ** beta

        eta_list = []
        beta_list = []
        for i in range(m):
            random_index = np.random.choice(np.arange(len(X_)), len(X_))
            random_X_ = X_[random_index]
            random_s_ = s_[random_index]
            random_V_ = V_[random_index]
            random_h_ = h_[random_index]
            new_params, error = self.optimization(random_X_, random_h_, random_s_, random_V_)
            eta_list.append(new_params[0])
            beta_list.append(new_params[1])

        return eta_list, beta_list


    def get_t_statistic(self, eta_list, beta_list):
        SE_eta = np.sqrt(np.var(eta_list))
        SE_beta = np.sqrt(np.var(beta_list))
        t_statistic_eta = np.average(eta_list) / SE_eta
        t_statistic_beta = np.average(beta_list) / SE_beta
        print("eta t_statistic equals {}".format(t_statistic_eta))
        print("beta t_statistic equals {}".format(t_statistic_beta))

        return t_statistic_eta, t_statistic_beta

    def compare_params_by_liquidity(self, k=2):
        less_active = list(self.total_volume.mean().sort_values()[:k].index)
        more_active = list(self.total_volume.mean().sort_values()[-k:].index)

        print("less active stocks have following eta and beta:")
        X_low, h_low, s_low, V_low = self.preprocessing(ticker_list=less_active,
                                                        splitting_by_liquidity=True)
        self.optimization(X_low, h_low, s_low, V_low)

        print("more active stocks have following eta and beta:")
        X_high, h_high, s_high, V_high = self.preprocessing(ticker_list=more_active,
                                                        splitting_by_liquidity=True)
        self.optimization(X_high, h_high, s_high, V_high)

if __name__ == "__main__":
    TAQRegression_obj = TAQRegression()

    TAQRegression_obj.NLR()
    print()
    eta_1, beta_1 = TAQRegression_obj.bootstrapping_residual( 40)
    eta_2, beta_2 = TAQRegression_obj.bootstrapping_paired( 40)
    print("Residual bootstrapping method yields eta equals {} and beta equals {}".format(np.average(eta_1),
                                                                                         np.average(beta_1)))
    print("Paired bootstrapping method yields eta equals {} and beta equals {}".format(np.average(eta_2),
                                                                                       np.average(beta_2)))
    print()
    TAQRegression_obj.get_t_statistic(eta_1, beta_1)
    TAQRegression_obj.get_t_statistic(eta_2, beta_2)

    print()
    TAQRegression_obj.compare_params_by_liquidity()