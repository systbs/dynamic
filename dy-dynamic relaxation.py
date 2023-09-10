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
M = 0.5 * K
W = np.sqrt(2)
X = np.array([[0.5, 0.5]]).T

while np.abs(prev_cost - cost) > epsilon and max_epoch > epoch:
    P = np.array([[yi for _,yi in data_points]]).T
    F = np.dot(K, X)
    R = (P - F)

    invM = np.linalg.pinv(M)

    A = - eta * (1/(W**2)) * np.dot(invM, R) * np.cos(W)
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

print(f"Time running: {total_time} s")

plt.plot([x for x,_ in data_result], [y for _,y in data_result],marker='o')
plt.xlabel('epoch')
plt.ylabel('Cost')
plt.show()