import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import statsmodels.api as sm


frame = pd.read_csv("./bluetooth/bt_u01.csv",nrows = 20)
#frame = pd.read_csv("./bluetooth/bt_u01.csv")
bt_data = frame.to_numpy()
print(len(frame))
print(frame)
'''
x = frame.to_numpy()
data = frame.drop_duplicates(subset=['Latitude'], keep='first', inplace=False)
data = data.reset_index(drop=True)
data.to_csv('./latitude.csv', encoding='utf8')
latitude_norepeat = data.to_numpy()
'''
level = []
time = []

for i in range (len(bt_data)):
    level.append(bt_data[i][3])
    time.append(bt_data[i][0])

bt_level = np.hstack((time,level))

#plt.plot(latitude,longitude)
#plt.plot(longitude, latitude)
'''
plt.plot(time,level)
plt.title("time vs level")
plt.xlabel("time")
plt.ylabel("level")
plt.legend()
plt.show()
'''
sm.graphics.tsa.plot_acf(bt_level)
#plt.figure(figsize=(12, 6))
plt.title("ACF for time lags vs bluetooth level (dataset size:100)")
plt.xlabel("time lag")
plt.ylabel("level")
plt.show()


sm.graphics.tsa.plot_pacf(bt_level)
plt.title("PACF for time lags vs bluetooth level (dataset size:100)")
plt.xlabel("time lag")
plt.ylabel("level")
plt.show()