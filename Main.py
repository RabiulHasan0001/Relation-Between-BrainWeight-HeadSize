import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
plt.rcParams['figure.figsize']=(20.0,10.0)

#Reading te data
data=pd.read_csv('headbrain.csv')
print(data.head)

#Collecting x and y
x=data['Head Size(cm^3)'].values
y=data['Brain Weight(grams)'].values
print(x)
print(y)

# Finding mean of x and y
x_mean=np.mean(x)
y_mean=np.mean(y)

#Total number of values
n=len('x')

#Using the formula
num=0
denom=0
for i in range(n):
    num+=(x[i]-x_mean)*(y[i]-y_mean)
    denom+=(x[i]-x_mean)**2

b1=(num)/(denom)
c1=(y_mean)-(b1*x_mean)    
print(b1,c1)

##Plotting graph
max_x=np.max(x)+100
min_x=np.min(x)-100
print(max_x)
print(min_x)

#Calculating line values  x and y
x=np.linspace(min_x,max_x,1000)
y=b1*x+c1
#Plotting line
plt.plot(x,y,color='#58b970',label='Regression line')
plt.scatter(x,y,c='#ef5423',label='Scatter plot')
plt.xlabel('Head size in cm3')
plt.ylabel('Brain weight in grem')
plt.legend()
plt.show()

#
p=0
q=0

for i in range(n):
    ypred=b1*x[i] + c1
    p+=(ypred-y[i])**2
    q+=(y[i]-y_mean)**2

r2=1-(p/q)
print(r2)    

##We can make it short using scikit without writing ormulas brifly
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
x=x.reshape((n,1))
reg=LinearRegression()
reg=reg.fit(x,y)
y_pred=reg.predict(x)
r2_score=reg.score(x,y)
print(r2_score)