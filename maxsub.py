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
#frame = pd.read_csv("./audio/audio_u00.csv",skiprows=854360, nrows = 5)
frame = pd.read_csv("./audio/audio_u02.csv")
bt_data = frame.to_numpy()
print(len(frame))
#print(frame)
level = []
time = []
start  = 0
end = 0
count = 0
max = 0

for i in range (len(bt_data)):
    level.append(bt_data[i][1])
    #if (i < )
    time.append(bt_data[i][0])

#print(frame.iloc[854361])
#print(frame.iloc[2549557])

each_value,each_value_count = np.unique(level,return_counts = True)
print(each_value)
print(each_value_count)

j = 0
while j < len(bt_data):
    if (level[j] == 0):
        #print("j",j)
        k=j
        count = 0
        while k<len(bt_data) and level[k] == 0:
            #print("k" ,k, "level ", level[k])
            count +=  1
            k+=1
        if(max < count):
            max = count
            #start = time[j]
            #end = time[j+count-1]
            start = j
            end = k
        j=k+1  
        #count = 0
    else: j += 1
print("Length:" ,max)
print("Start position:",start, "Start time: ", time[start])
print("End position:" ,end, "End time: ", time[end])


print(frame.iloc[start-1:end+1])

each_value_test,each_value_count_test = np.unique(frame.iloc[start-1:end+1,1:2],return_counts = True)
print(each_value_test)
print(each_value_count_test)