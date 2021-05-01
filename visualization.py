# create and evaluate a static autoregressive model
from pandas import read_csv
from datetime import datetime

from statsmodels.tsa.ar_model import AutoReg
from sklearn.metrics import mean_squared_error
from math import sqrt
import numpy as np
import pandas_datareader as pdr
import pandas as pd
import seaborn as sns
from statsmodels.tsa.ar_model import AutoReg, ar_select_order
from statsmodels.tsa.api import acf, pacf, graphics
import matplotlib.pyplot as plt
# load dataset
df = pd.read_csv('./dataset/dataset/sensing/audio/audio_u00.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')

series = df[:25484]
print(series)

series.plot(kind='scatter',x='timestamp',y=' audio inference',color='red')
plt.show()
