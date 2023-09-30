import numpy as np
import matplotlib.pyplot as plt
import time
start_time = time.time()

# Define data points
data_points = [(1, 2), (2, 9), (3, 16), (4, 25), (5, 36)]

# Define algorithm parameters
learning_rate = 0.1
epsilon = 1e-8

# Initialize sum of squared gradients for each parameter
sum_squared_gradients = np.zeros(2)

# Initialize weights
weights = np.array([0.0, 0.0])

def model(w, x):
    return w[0] * x + w[1]

def mean_squared_error(w):
    total_error = 0
    for x, y in data_points:
        total_error += (model(w, x) - y) ** 2
    return total_error / len(data_points)

cost = 1
prev_cost = 2
epoch = 1

max_epoch = 5000

analysis = []

# Training loop for Adagrad
while np.abs(cost - prev_cost) > epsilon and max_epoch > epoch:
    for data_point in data_points:
        x, y = data_point
        
        # Calculate model prediction
        prediction = weights[0] * x + weights[1]
        
        # Calculate error
        error = prediction - y
        
        # Calculate gradients
        gradients = np.array([error * x, error])
        
        # Update sum of squared gradients
        sum_squared_gradients += gradients ** 2
        
        # Update weights using Adagrad
        weights -= learning_rate * gradients / (np.sqrt(sum_squared_gradients) + epsilon)

        cost = mean_squared_error(weights)

        analysis.append([epoch, cost])

        print(f"epoch:{epoch} cost:{cost}")

        epoch = epoch + 1

# Display final weights
print(f"Final weights:{weights}")

end_time = time.time()
total_time = end_time - start_time

print(f"Time running: {total_time} s")

plt.plot([x for x,_ in analysis], [y for _,y in analysis],marker='o')
plt.xlabel('epoch')
plt.ylabel('Cost')
plt.show()