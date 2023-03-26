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

    def NLR(self):
        X_ = self.vwap400 * self.market_imbalance
        h_ = self.temporary_impact
        s_ = self.return_std
        V_ = self.vwap400 * self.total_volume
        self.optimization(X_, h_, s_, V_)
        return

    def optimization(self, X_, h_, s_, V_):
        def curve_fit_func(X, eta, beta):
            X_, s_, V_ = X
            return eta * s_ * np.sign(X_) * (np.abs(X_) / ((6 / 6.5) * V_)) ** beta

        InitialParameter = (0.5, 0.5)

        # model_df = pd.DataFrame({"imbalance_value": X_,
        #                          "temp_impact": h_,
        #                          "daily_std": s_,
        #                          "daily_volume": V_})

        least_square_func = lambda p, x, y: y - p[0] * x["daily_std"] * np.sign(x["imbalance_value"]) * \
                                            (np.abs(x["imbalance_value"]) / ((6 / 6.5) * x["daily_volume"])) ** p[1]

        # curve fit the test data
        para1_list = []
        para2_list = []
        print(self.vwap400.index)
        for date in self.vwap400.index:
            X_ = self.vwap400.loc[date,:] * self.market_imbalance.loc[date,:]
            h_ = self.temporary_impact.loc[date,:]
            s_ = self.return_std.loc[date,:]
            V_ = self.vwap400.loc[date,:] * self.total_volume.loc[date,:]
            fittedParameters, pcov = curve_fit(curve_fit_func, (X_, s_, V_), h_, InitialParameter, bounds=(-1, 1.5))
            para1_list.append(fittedParameters[0])
            para2_list.append(fittedParameters[1])
        para1 = np.mean(para1_list)
        para2 = np.mean(para2_list)
        fittedParameters = [para1, para2]
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

        X_ = self.vwap400 * self.market_imbalance
        h_ = self.temporary_impact
        s_ = self.return_std
        V_ = self.vwap400 * self.total_volume

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

        X_ = self.vwap400 * self.market_imbalance
        h_ = self.temporary_impact
        s_ = self.return_std
        V_ = self.vwap400 * self.total_volume

        def curve_fit_func(X, eta, beta):
            X_, s_, V_ = X
            return eta * s_ * np.sign(X_) * (np.abs(X_) / ((6 / 6.5) * V_)) ** beta

        eta_list = []
        beta_list = []
        X_.reset_index(inplace = True)
        print(X_)
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

    # # def compare_params_by_liquidity(self, k=2):
    # #     less_active = list(self.total_volume.mean().sort_values()[:k].index)
    # #     more_active = list(self.total_volume.mean().sort_values()[-k:].index)
    # #
    # #     print("less active stocks have following eta and beta:")
    # #     X_low, h_low, s_low, V_low = self.preprocessing(ticker_list=less_active,
    # #                                                     splitting_by_liquidity=True)
    # #     self.optimization(X_low, h_low, s_low, V_low)
    # #
    # #     print("more active stocks have following eta and beta:")
    # #     X_high, h_high, s_high, V_high = self.preprocessing(ticker_list=more_active,
    # #                                                     splitting_by_liquidity=True)
    #     self.optimization(X_high, h_high, s_high, V_high)

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

    # print()
    # TAQRegression_obj.compare_params_by_liquidity()