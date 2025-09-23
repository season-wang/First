# y = wx + b
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

x_data = [1,2,3]
y_data = [3,5,7]
def forward(w , b , x):
    return w * x + b

def loss(w , b , x , y):
    y_pred = forward(w , b , x)
    return (y_pred - y) ** 2

w_list = []
b_list = []
MSE_list = []

for w in np.arange(0.0 , 5.1 , 1):
    for b in np.arange(0.0 , 5.1 , 0.1):
        w = round(w,1)
        b = round(b,1)
        print("w = " , w , "b = " , b)
        loss_sum = 0
        for x_val , y_val in zip(x_data , y_data):
            y_pred = forward(w , b , x_val)
            loss_val = loss(w , b , x_val , y_val)
            loss_sum += loss_val
            print('x\ty\ty_pred\tloss_val\n', x_val, '\t', y_val, '\t', round(y_pred,3), '\t\t', round(loss_val,3))
        print('MSE = ', round(loss_sum / len(x_data),3))
        print('\n')
        w_list.append(w)
        b_list.append(b)
        MSE_list.append(round(loss_sum / len(x_data),3))


wb = list(zip(w_list , b_list))

w0b = []
for tup in wb:
    if tup[0] == 0:
        w0b.append(tup[1])

w1b = []
for tup in wb:
    if tup[0] == 0:
        w1b.append(tup[1])

w2b = []
for tup in wb:
    if tup[0] == 0:
        w2b.append(tup[1])

w3b = []
for tup in wb:
    if tup[0] == 0:
        w3b.append(tup[1])

w4b = []
for tup in wb:
    if tup[0] == 0:
        w4b.append(tup[1])

w5b = []
for tup in wb:
    if tup[0] == 0:
        w5b.append(tup[1])

step = 51

plt.figure(figsize=(10,8))

plt.title('MSE vs w=0,1,2,3,4,5')

plt.plot(w0b, MSE_list[0:step],'b',label = 'W = 0, b = 0 ,0.1...5.0')
plt.plot(w1b, MSE_list[step:2*step],'r',label = 'W = 1, b = 0 ,0.1...5.0')
plt.plot(w2b, MSE_list[2*step:3*step],'g',label = 'W = 2, b = 0 ,0.1...5.0')
plt.plot(w3b, MSE_list[3*step:4*step],'y',label = 'W = 3, b = 0 ,0.1...5.0')
plt.plot(w4b, MSE_list[4*step:5*step],'m',label = 'W = 4, b = 0 ,0.1...5.0')
plt.plot(w5b, MSE_list[5*step:6*step],'black',label = 'W = 5, b = 0 ,0.1...5.0')

plt.xlabel('b value')
plt.ylabel('MSE value')
plt.legend()
plt.show()


