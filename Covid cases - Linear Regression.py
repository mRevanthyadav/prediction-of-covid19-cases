import pandas as pd

import numpy as np

from sklearn.linear_model import LinearRegression 
import matplotlib.pyplot as plt

df = pd.read_csv('1rdata.csv') 
df.head()

y = df['time']

x = df[['cases', 'increasingrate', 'decreasingrate']]

import pandas as pd
import numpy as np from sklearn.linear_model
import LinearRegression
import matplotlib.pyplot as plt

df = pd.read_csv('Irdata.csv')

df.head()

y = df['time']

x = df [[ 'cases', 'increasingrate', 'decreasingrate']]

def CasesReg(x,y):

from sklearn.linear_model import LinearRegression
reg=LinearRegression()

reg.fit(x,y)

return reg

reg = CasesReg(x,y)

y_pred = reg.predict(x)

y_pred

plt.plot(x,reg.predict(x),'*')

plt.legend (labels=['cases', 'increasingrate', 'decreasingrate']) 

plt.xlabel("Cases Rate")

plt.ylabel("Time")

plt.show()