import numpy as np
import matplotlib.pyplot as plt
import time
start_time = time.time()

# Input data points
data_points = [(1, 2), (2, 9), (3, 16), (4, 25), (5, 36)]

# Our model: y = wx + b (linear)
def model(w, x, b):
    return w * x + b

# Cost function: Mean Squared Error
def mean_squared_error(w, b):
    total_error = 0
    for x, y in data_points:
        total_error += (model(w, x, b) - y) ** 2
    return total_error / len(data_points)

# Compute gradients of the cost function with respect to weights and bias
def compute_gradients(w, b):
    dw = 0
    db = 0
    for x, y in data_points:
        dw += 2 * (model(w, x, b) - y) * x
        db += 2 * (model(w, x, b) - y)
    dw /= len(data_points)
    db /= len(data_points)
    return dw, db

# Initial values for weights and bias
w = 0.5
b = 0.5

# Learning rate (step size for gradient descent)
learning_rate = 0.01

# Number of epochs for gradient descent
prev_cost = 2
cost = 1
epoch = 1
max_epoch = 5000

data_result = []

# Training the model using gradient descent
while np.abs(prev_cost-cost) > 1e-8 and max_epoch > epoch:
    # Compute gradients
    dw, db = compute_gradients(w, b)
    
    # Update weights and bias
    w = w - learning_rate * dw
    b = b - learning_rate * db
    
    # Print the cost value at each epoch (to show the cost reduction)
    prev_cost = cost
    cost = mean_squared_error(w, b)

    data_result.append([epoch, cost])

    print(f"epoch = {epoch} cost = {cost}")
    epoch = epoch + 1

# Sample the final weights and bias after training
print("Model parameters:")
print([w, b])

end_time = time.time()
total_time = end_time - start_time

print(f"Time running: {total_time} s")

plt.plot([x for x,_ in data_result], [y for _,y in data_result],marker='o')
plt.xlabel('epoch')
plt.ylabel('Cost')
plt.show()
