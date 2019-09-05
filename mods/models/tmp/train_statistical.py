# -*- coding: utf-8 -*-
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""
Created on 28-05-2019

Time Series Forecasting: statistical methods

@author: giangnguyen

modified from https://machinelearningmastery.com/time-series-forecasting-methods-in-python-cheat-sheet/
"""

import mods.config as cfg
import mods.utils as utl

import datetime

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

from statsmodels.tsa.ar_model import AR
from statsmodels.tsa.arima_model import ARMA
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.tsa.statespace.varmax import VARMAX
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from random import random

def create_random_dataset(model_type):
    data = []
    data_exog = []
    if model_type in ['VAR', 'VARMA', 'VARMAX']:
        for i in range(100):
            v1 = i + random()
            v2 = v1 + random()
            data.append([v1, v2])
        data_exog = [x + random() for x in range(100)]
    else:
        data = [x + random() for x in range(1, 100)]
        if model_type == 'SARIMAX':
            data_exog = [x + random() for x in range(101, 200)]
    return data, data_exog


def train(model_type, data, data_exog=[]):

    if model_type == 'AR':                  # AutoRegression (AR)
        model = AR(data)
        model_fit = model.fit()
    elif model_type == 'MA':                # Moving Average (MA)
        model = ARMA(data, order=(0, 1))
        model_fit = model.fit(disp=False)
    elif model_type == 'ARMA':              # AutoRegressive Moving Average (ARMA)
        model = ARMA(data, order=(2, 1))
        model_fit = model.fit(disp=False)
    elif model_type == 'ARIMA':             # Autoregressive Integrated Moving Average (ARIMA)
        model = ARIMA(data, order=(1, 1, 1))
        model_fit = model.fit(disp=False)
    elif model_type == 'SARIMA':            # Seasonal Autoregressive Integrated Moving-Average (SARIMA)
        model = SARIMAX(data, order=(1, 1, 1), seasonal_order=(1, 1, 1, 1))
        model_fit = model.fit(disp=False)
    elif model_type == 'VAR':               # Vector AutoRegression (VAR)
        model = VAR(data)
        model_fit = model.fit()
    elif model_type == 'VARMA':             # Vector AutoRegression Moving-Average (VARMA)
        model = VARMAX(data, order=(1, 1))
        model_fit = model.fit(disp=False)
    elif model_type == 'SES':               # Simple Exponential Smoothing (SES)
        model = SimpleExpSmoothing(data)
        model_fit = model.fit()
    else:                                   # Holt Winter’s Exponential Smoothing (HWES)
        model = ExponentialSmoothing(data)
        model_fit = model.fit()

    if data_exog:
        if model_type == 'SARIMAX':         # Seasonal Autoregressive Integrated Moving-Average with Exogenous Regressors (SARIMAX)
            model = SARIMAX(data, exog=data_exog, order=(1, 1, 1), seasonal_order=(0, 0, 0, 0))
            model_fit = model.fit(disp=False)
        elif model_type == 'VARMAX':        # Vector AutoRegression Moving-Average with Exogenous Regressors (VARMAX)
            model = VARMAX(data, exog=data_exog, order=(1, 1))
            model_fit = model.fit(disp=False)

    return model_fit


def predict(model_type, model_fit, data):

    if model_type == 'ARIMA':
        yhat = model_fit.predict(len(data), len(data), typ='levels')
    elif model_type == 'SARIMAX':
        data_exog2 = [200 + random()]
        yhat = model_fit.predict(len(data), len(data), exog=[data_exog2])
    elif model_type == 'VAR':
        yhat = model_fit.forecast(model_fit.y, steps=1)
    elif model_type == 'VARMA':
        yhat = model_fit.forecast()
    elif model_type == 'VARMAX':
        data_exog2 = [[100]]
        yhat = model_fit.forecast(exog=data_exog2)
    else:
        yhat = model_fit.predict(len(data), len(data))
    print(model_type, yhat)

    return yhat


def eval():
    # mae = mean_absolute_error(data, yhat)
    # mse = mean_squared_error(data, yhat)
    # smape = utl.smape(data, yhat)
    # print(model_type, '\t', 'mae=', mae, '\t', 'mse=', mse, '\t', 'smape=', smape)
    return


if __name__ == "__main__":
    # print(cfg.BASE_DIR)

    # timestamp
    timer_start = datetime.datetime.now()
    ts = str(timer_start).replace(' ', '_').replace('-', '').replace(':', '').split('.')[0]
    eval_fn = cfg.app_data_results + ts + '-' + cfg.eval_filename

    # model_types = ['AR', 'MA', 'ARMA', 'ARIMA', 'SARIMA', 'SARIMAX', 'VAR', 'VARMA', 'VARMAX', 'SES', 'HWES']
    model_types = ['AR', 'MA', 'ARIMA', 'SARIMA', 'SES', 'HWES']

    for model_type in model_types:
        data, data_exog = create_random_dataset(model_type)
        model_fit = train(model_type, data, data_exog)
        predict(model_type, model_fit, data)

    # print(eval_fn)
    # runtime report
    print('\nRuntime =', datetime.datetime.now() - timer_start)