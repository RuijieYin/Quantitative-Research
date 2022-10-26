# import basics
import datetime
import math
import time
from datetime import timedelta
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib
from matplotlib import pyplot as plt

# import from sklearn
import sklearn
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

# import from joinquant
from jqlib.technical_analysis import *
from jqfactor import Factor, calc_factors
from jqfactor import get_factor_values
from jqfactor import neutralize
from jqfactor import winsorize
from jqfactor import standardlize

# some of the relevant key factors (defined in Joinquant)
factors = ['operating_revenue_growth_rate', 'total_asset_growth_rate', 'net_operate_cashflow_growth_rate',
           'total_profit_growth_rate', 'net_profit_growth_rate', 'total_operating_revenue_ttm',
           'operating_profit_ttm', 'gross_profit_ttm', 'EBIT',
           'net_profit_ttm', 'market_cap', 'cash_flow_to_price_ratio',
           'sales_to_price_ratio', 'net_profit_ratio', 'quick_ratio',
           'current_ratio', 'operating_profit_ratio', 'SGI',
           'roe_ttm', 'VOL10', 'VOL20', 'VOL60', 'VOL120',
           'AR', 'BR', 'ARBR', 'VEMA5', 'DAVOL10', 'DAVOL5', 'Variance20',
           'Variance60', 'Variance120', 'ATR6', 'ATR14', 'total_operating_revenue_per_share',
           'eps_ttm', 'BIAS10', 'BIAS20', 'BIAS60', 'Volume1M',
           'momentum', 'book_to_price_ratio', 'liquidity', 'earnings_yield',
           'growth']

# parameters:
data_input_user =
start_date_user = "2019-01-01"
end_date_user = "2022-09-15"


# define log of the closing price:
class LOG_CLOSE(Factor):
    name = 'log_close'
    max_window = 1
    dependencies = ['close']

    def calc(self, data):
        log_close = np.log(data['close'])
        return log_close.mean()


# define EP
class EP(Factor):
    name = 'ep'
    max_window = 1
    dependencies = ['net_profit_ttm', 'market_cap']  # net profit/market cap

    def calc(self, data):
        total_net_profit = data['net_profit_ttm']
        market_value = data['market_cap']
        EP = total_net_profit / market_value

        return EP.mean()


# define BP
class BP(Factor):
    name = "bp"
    max_window = 1
    dependencies = ['total_assets', 'total_liability', 'market_cap']

    def calc(self, data):
        net_asset = data['total_assets'] - data['total_liability']
        bp = net_asset / data['market_cap']

        return bp.mean()


# gross profit:
class GROSSPROFITABILITY(Factor):
    name = 'gross_profitability'
    max_window = 1
    dependencies = ['total_operating_revenue', 'total_operating_cost', 'total_assets']

    def calc(self, data):
        total_operating_revenue = data['total_operating_revenue']
        total_operating_cost = data['total_operating_cost']
        total_assets = data['total_assets']
        gross_profitability = (total_operating_revenue - total_operating_cost) / total_assets
        return gross_profitability.mean()


# ROATTM
class ROATTM(Factor):
    name = 'roa_ttm'
    max_window = 1
    dependencies = ['net_profit', 'net_profit_1', 'net_profit_2', 'net_profit_3',
                    'total_assets']

    def calc(self, data):
        net_profit_ttm = data['net_profit'] + data['net_profit_1'] + data['net_profit_2'] + data['net_profit_3']
        result = net_profit_ttm / data['total_assets']
        return result.mean()


# SP
class SP(Factor):
    name = 'sp'
    max_window = 1
    dependencies = ['operating_revenue_ttm', 'market_cap']

    def calc(self, data):
        SP = data['operating_revenue_ttm'] / data['market_cap']
        return SP.mean()


# increment in profit
class INCPROFITYOY(Factor):
    name = 'profityoy'
    max_window = 1
    dependencies = ['inc_net_profit_year_on_year']

    def calc(self, data):
        yoy = data['inc_net_profit_year_on_year']
        return yoy.mean()


# increment in revenue
class INCREVENUEYOY(Factor):
    name = 'revenueyoy'
    max_window = 1
    dependencies = ['inc_revenue_year_on_year']

    def calc(self, data):
        yoy = data['inc_revenue_year_on_year']
        return yoy.mean()


# increment in ROA
class ROA_GRO2(Factor):
    name = 'roa_gro2'
    max_window = 1
    dependencies = ['roa_y', 'roa_y1']

    def calc(self, data):
        gro = (data['roa_y'] - data['roa_y1']) / data['roa_y1']  # calculate ROA

        return gro.mean()


# increment in net profit:
class PROFIT_GRO(Factor):
    name = 'profit_gro'
    max_window = 1
    dependencies = ['net_profit_y', 'net_profit_y1']

    def calc(self, data):
        gro = (data['net_profit_y'] - data['net_profit_y1']) / data['net_profit_y1']

        return gro.mean()


# increment in sales
class SALES_GRO(Factor):
    name = 'sales_gro'
    max_window = 1
    dependencies = ['operating_revenue', 'operating_revenue_y1']

    def calc(self, data):
        gro = (data['operating_revenue'] - data['operating_revenue_y1']) / data['operating_revenue_y1']

        return gro.mean()


# net operational cash flow/net profit
class OPERATIONCASHRATIO_Q(Factor):
    name = 'operationcashratio_q'
    max_window = 1
    dependencies = ['net_operate_cash_flow', 'net_profit']

    def calc(self, data):
        factor = data['net_operate_cash_flow'] / data['net_profit']

        return factor.mean()


# net operational cash flow/TTM
class OPERATIONCASHRATIO_TTM(Factor):
    name = 'operationcashratio_ttm'
    max_window = 1
    dependencies = ['net_operate_cash_flow_ttm', 'net_profit',
                    'net_profit_1', 'net_profit_2', 'net_profit_3']

    def calc(self, data):
        net_profit_ttm = data['net_profit'] + data['net_profit_1'] + data['net_profit_2'] + data['net_profit_3']
        factor = data['net_operate_cash_flow_ttm'] / net_profit_ttm

        return factor.mean()


# ROA
class ROA(Factor):
    name = 'roa'
    max_window = 1
    dependencies = ['roa']

    def calc(self, data):
        roa = data['roa']
        return roa.mean()


# ROE
class ROE(Factor):
    name = 'roe'
    max_window = 1
    dependencies = ['roe']

    def calc(self, data):
        roe = data['roe']
        return roe.mean()


# net profit
class NET_PRO(Factor):
    name = 'net_pro'
    max_window = 1
    dependencies = ['net_profit']

    def calc(self, data):
        net_profit = data['net_profit']
        return net_profit.mean()


# asset turnover rate
class ASSETTURNOVER_Q(Factor):
    name = 'assetturnover_q'
    max_window = 1
    dependencies = ['operating_revenue', 'total_assets']

    def calc(self, data):
        turnover = data['operating_revenue'] / data['total_assets']
        return turnover.mean()


# YTD
class PROFITMARGIN_Q(Factor):
    name = 'profitmargin_q'
    max_window = 1
    dependencies = ['gross_profit_margin']

    def calc(self, data):
        profitm = data['gross_profit_margin']
        return profitm.mean()


# total asset/net asset
class FINANCIAL_LEVERAGE(Factor):
    name = 'financial_leverage'
    max_window = 1
    dependencies = ['total_assets', 'total_liability']

    def calc(self, data):
        net_asset = data['total_assets'] - data['total_liability']
        leverage = data['total_assets'] / net_asset

        return leverage.mean()


# leverage/net asset
class DEBTEQUITY_RATIO(Factor):
    name = 'debtequity_ratio'
    max_window = 1
    dependencies = ['total_assets', 'total_liability', 'total_non_current_liability']

    def calc(self, data):
        net_asset = data['total_assets'] - data['total_liability']
        leverage = data['total_non_current_liability'] / net_asset

        return leverage.mean()


# yields in 180 days/360 days

class ALPHA_180(Factor):
    name = 'alpha_180'
    max_window = 181
    dependencies = ['close']

    def calc(self, data):
        close = data['close'].pct_change()[1:]
        index_close = self._get_extra_data(securities=['000001.XSHG'], fields=['close'])['close'].pct_change()[1:]
        # regression
        model = sm.OLS(close, sm.add_constant(index_close.values))
        result = model.fit()
        alpha = result.params[0]

        return alpha.mean()


class ALPHA_360(Factor):
    name = 'alpha_360'
    max_window = 361
    dependencies = ['close']

    def calc(self, data):
        close = data['close'].pct_change()[1:]
        index_close = self._get_extra_data(securities=['000001.XSHG'], fields=['close'])['close'].pct_change()[1:]
        # regression
        model = sm.OLS(close, sm.add_constant(index_close.values))
        result = model.fit()
        alpha = result.params[0]

        return alpha.mean()


class BETA_180(Factor):
    name = 'beta_180'
    max_window = 181
    dependencies = ['close']

    def calc(self, data):
        close = data['close'].pct_change()[1:]
        index_close = self._get_extra_data(securities=['000001.XSHG'], fields=['close'])['close'].pct_change()[1:]
        # regression
        model = sm.OLS(close, sm.add_constant(index_close.values))
        result = model.fit()
        beta = result.params[1]

        return beta.mean()


class BETA_360(Factor):
    name = 'beta_360'
    max_window = 361
    dependencies = ['close']

    def calc(self, data):
        close = data['close'].pct_change()[1:]
        index_close = self._get_extra_data(securities=['000001.XSHG'], fields=['close'])['close'].pct_change()[1:]
        # regression
        model = sm.OLS(close, sm.add_constant(index_close.values))
        result = model.fit()
        beta = result.params[1]

        return beta.mean()


# yields in the past 30,60,180,240 days
class RET_30(Factor):
    name = 'ret_30'
    max_window = 30
    dependencies = ['close']

    def calc(self, data):
        close = data['close']
        ret = close.iloc[-1, :] / close.iloc[0, :] - 1
        return ret


class RET_60(Factor):
    name = 'ret_60'
    max_window = 60
    dependencies = ['close']

    def calc(self, data):
        close = data['close']
        ret = close.iloc[-1, :] / close.iloc[0, :] - 1
        return ret


class RET_180(Factor):
    name = 'ret_180'
    max_window = 180
    dependencies = ['close']

    def calc(self, data):
        close = data['close']
        ret = close.iloc[-1, :] / close.iloc[0, :] - 1
        return ret


class RET_360(Factor):
    name = 'ret_360'
    max_window = 360
    dependencies = ['close']

    def calc(self, data):
        close = data['close']
        ret = close.iloc[-1, :] / close.iloc[0, :] - 1
        return ret


# turnover rate in the past 30,60,180,240 days
class TURN_RET30(Factor):
    name = 'turn_ret30'
    max_window = 31
    dependencies = ['volume', 'circulating_cap', 'close']

    def calc(self, data):
        volume = data['volume'] / 10000
        turn = volume / data['circulating_cap']
        turn = turn[1:]
        close = data['close'].pct_change()[1:]
        turn_ret = close * turn

        return turn_ret.mean()


class TURN_RET60(Factor):
    name = 'turn_ret60'
    max_window = 61
    dependencies = ['volume', 'circulating_cap', 'close']

    def calc(self, data):
        volume = data['volume'] / 10000
        turn = volume / data['circulating_cap']
        turn = turn[1:]
        close = data['close'].pct_change()[1:]
        turn_ret = close * turn

        return turn_ret.mean()


class TURN_RET120(Factor):
    name = 'turn_ret120'
    max_window = 121
    dependencies = ['volume', 'circulating_cap', 'close']

    def calc(self, data):
        volume = data['volume'] / 10000
        turn = volume / data['circulating_cap']
        turn = turn[1:]
        close = data['close'].pct_change()[1:]
        turn_ret = close * turn

        return turn_ret.mean()


class TURN_RET180(Factor):
    name = 'turn_ret180'
    max_window = 181
    dependencies = ['volume', 'circulating_cap', 'close']

    def calc(self, data):
        volume = data['volume'] / 10000
        turn = volume / data['circulating_cap']
        turn = turn[1:]
        close = data['close'].pct_change()[1:]
        turn_ret = close * turn

        return turn_ret.mean()


# return a panel of weights of these factors
stock_list = list(get_index_weights(data_input_user, date='a due date').index)
factor_data = get_factor_values(securities=stock_list,
                                factors=factors,
                                start_date=start_date_user, end_date=end_date_user)
df_factor = pd.DataFrame()
factor_name = list(factor_data.keys())
for name in factor_name:
    df_factor = pd.concat([df_factor, factor_data[name]])
df_factor = df_factor.T
df_factor.columns = factor_name
# check if there are missing values
print(df_factor.isnull().sum())
