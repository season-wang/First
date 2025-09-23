# y = wx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


x_data = [1,2,3]
y_data = [2,4,6]

def forward(w , x):
    return w * x

def loss(w , x , y):
    y_pred = forward(w , x)
    return (y_pred - y) ** 2

w_list = []
MSE_list = []

for w in np.arange(0.0 , 5.1 , 0.1):
    w = round(w,1)
    print("w = " , w)
    loss_sum = 0
    for x_val , y_val in zip(x_data , y_data):
        y_pred = forward(w , x_val)
        loss_val = loss(w , x_val , y_val)
        loss_sum += loss_val
        print('x\ty\ty_pred\tloss_val\n' , x_val ,'\t', y_val ,'\t' ,  round(y_pred) ,'\t\t' ,  round(loss_val))
    print('MSE = ' , round(loss_sum / len(x_data)))
    print('\n')
    w_list.append(w)
    MSE_list.append(round(loss_sum / len(x_data)))

plt.title('relationship with w and MSE')
plt.plot(w_list , MSE_list)
plt.show()

