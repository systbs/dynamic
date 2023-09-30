# dynamic
Dynamic Optimization Method

# method
For example:

P=α0+α1x1+α2x2

Then:

X= [α0,α1,α2]

If data point is: 

` `data = [(a<sub>1</sub>,a<sub>2</sub>,p<sub>1</sub>), (a<sub>3</sub>,a<sub>4</sub>,p<sub>2</sub>),…, (a<sub>5</sub>,a<sub>6</sub>,p<sub>3</sub>), (a<sub>7</sub>,a<sub>8</sub>,p<sub>4</sub>)]

K=[1,x1,x2] → S=[[1,a1,a2],[1,a3,a4],[1,a5,a6],...[1,a7,a8]] 

P=(S^T)X
T is transpose

x[n+1]=x[n] + η A cos(ωt)

Where:

A=-(R/Mω^2)cos(ωt)=-(R/S^T)cos(ωt)

M = S/2

ω=sqrt(S/M)=sqrt(2)

cost = (R^T).R/n
n is number of inputs


# code
```python
# Input data points
import numpy as np
import matplotlib.pyplot as plt
import time
start_time = time.time()

data_points = [(1, 2), (2, 9), (3, 16), (4, 25), (5, 36)]

epsilon = 1e-8
eta = 42

data_result = []

epoch = 0
cost = 1
prev_cost = 2
max_epoch = 5000

K = np.array([[1, xi] for xi,_ in data_points])
W = np.sqrt(2)

P = np.array([[yi for _,yi in data_points]]).T
invK = np.linalg.pinv(K)
X = np.dot(invK, P)

while np.abs(prev_cost - cost) > epsilon and max_epoch > epoch:
    F = np.dot(K, X)
    R = (P - F)

    A = - eta * np.dot(invK, R) * np.cos(W)
    X = X - A * np.cos(W)

    prev_cost = cost
    cost = np.dot(R.T,R) / len(R)

    data_result.append([epoch, cost[0]])

    print(f"epoch={epoch} cost={cost}")

    epoch = epoch + 1
    
# Sample the final weights and bias after training
print(f"Final Weight: {X}")

end_time = time.time()
total_time = end_time - start_time

print(f"Time Running: {total_time} s")

plt.plot([x for x,_ in data_result], [y for _,y in data_result],marker='o')
plt.xlabel('Epoch')
plt.ylabel('Cost')
plt.show()
```
