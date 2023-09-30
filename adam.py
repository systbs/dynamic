import numpy as np
import matplotlib.pyplot as plt
import time
start_time = time.time()

# Input data points
data_points = [(1, 2), (2, 9), (3, 16), (4, 25), (5, 36)]
X = np.array([x for x, y in data_points])
y = np.array([y for x, y in data_points])

data_result = []

# Add a bias column to the input data for the constant term in the linear regression model
X_with_bias = np.c_[np.ones(X.shape[0]), X]

# Number of samples
m = X_with_bias.shape[0]

# Number of features
n_features = X_with_bias.shape[1]

# Setting parameters for the Adam algorithm
learning_rate = 0.1
beta1 = 0.9
beta2 = 0.999
epsilon = 1e-8

# Define parameters
theta = np.random.randn(n_features)

# Define variables for computing averages
m_t = np.zeros(n_features)
v_t = np.zeros(n_features)

def model(w, x):
    return w[1] * x + w[0]

def mean_squared_error(w):
    total_error = 0
    for x, y in data_points:
        total_error += (model(w, x) - y) ** 2
    return total_error / len(data_points)

# Training the model using the Adam algorithm
prev_cost = 2
cost = 1
epoch = 1
max_epoch = 5000

while np.abs(prev_cost-cost) > 1e-8 and max_epoch > epoch:
    # Compute gradients of the parameters
    gradients = (1/m) * X_with_bias.T.dot(X_with_bias.dot(theta) - y)
    
    # Update variables based on Adam
    m_t = beta1 * m_t + (1 - beta1) * gradients
    v_t = beta2 * v_t + (1 - beta2) * gradients**2
    m_t_hat = m_t / (1 - beta1**(epoch + 1))
    v_t_hat = v_t / (1 - beta2**(epoch + 1))
    
    theta -= learning_rate * m_t_hat / (np.sqrt(v_t_hat) + epsilon)

    prev_cost = cost
    cost = mean_squared_error(theta)

    data_result.append([epoch, cost])

    print(f"epoch = {epoch} cost = {cost}")
    epoch = epoch + 1

print("Model parameters:")
print(theta)

end_time = time.time()
total_time = end_time - start_time

print(f"Time running: {total_time} s")

plt.plot([x for x,_ in data_result], [y for _,y in data_result],marker='o')
plt.xlabel('epoch')
plt.ylabel('Cost')
plt.show()