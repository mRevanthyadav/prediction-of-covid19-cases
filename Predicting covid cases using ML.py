import numpy import random

import matplotlib.pyplot as plt

def CasesReg(cases_train, time_train):

from sklearn.linear_model import LinearRegression

reg Linear Regression()

reg.fit(cases train, time_train)

return reg

random.seed(42)

numpy.random.seed(42)

cases = []

time = []

for ii in range(100):

cases.append(random.randint(28,65))

time = [ii*6.25 + numpy.random.normal(scale=40.) for ii in cases] 
### need massage list into a 2d numpy array to get it to work in Linear Regression 
cases = numpy.reshape( numpy.array(cases), (len(cases), 1)) 
time numpy.reshape( numpy.array(time), (len(time), 1))

from sklearn.model_selection import train_test_split cases_train, cases_test, time train, time_test train_test_split(cases, time)

reg CasesReg(cases_train, time_train)

print("Slope", reg.coef_)

print("Intercept", reg.intercept_)

print("Testing data", reg.score (cases_test, time_test))

plt.plot(cases_test,reg.predict(cases_test)) 

plt.xlabel("Cases")

plt.ylabel("Time")

plt.show()