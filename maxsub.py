# create and evaluate a static autoregressive model
from pandas import read_csv
from datetime import datetime
import numpy as np
import pandas as pd


import matplotlib.pyplot as plt
# load dataset
#frame = pd.read_csv("./audio/audio_u00.csv",skiprows=854360, nrows = 5)
frame = pd.read_csv("./audio/audio_u02.csv") #the path of the csvfile
bt_data = frame.to_numpy()
print("csv file length:",len(frame))
#print(frame)
level = []      #storing the audio inference
time = []       #storing the time
start  = 0
end = 0
count = 0
max = 0

for i in range (len(bt_data)):
    level.append(bt_data[i][1])
    time.append(bt_data[i][0])


each_value,each_value_count = np.unique(level,return_counts = True)
print("all levels:", each_value)            #all levels
print("l number:",each_value_count)         #the number of each level
print()

j = 0
while j < len(bt_data):
    if (level[j] == 1):             #the user need to change this part as the audio inference you want
        #print("j",j)
        k=j
        count = 0
        while k<len(bt_data) and level[k] == 1:         #the user need to change this part as the audio inference you want
            #print("k" ,k, "level ", level[k])
            count +=  1
            k+=1
        if(max < count):
            max = count
            start = j       #the index of the start
            end = k         #the index of the end
        j=k+1  
    else: j += 1
print("Length:" ,max)
print("Start position:",start, "Start time: ", time[start])
print("End position:" ,end, "End time: ", time[end])


print()
print("The test part:")
print(frame.iloc[start-1:end+1])        #The data in the period we get

#the distribution of the level in this period
each_value_test,each_value_count_test = np.unique(frame.iloc[start-1:end+1,1:2],return_counts = True)
print(each_value_test)
print(each_value_count_test)