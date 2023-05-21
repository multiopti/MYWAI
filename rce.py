import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


def rce_train(class1, class2, eps, lambda_max):
    # Find number of train patterns (columns)
    n_c1p = class1.shape[0]
    n_c2p = class2.shape[0]

    lambda_1 = np.zeros(n_c1p)
    lambda_2 = np.zeros(n_c2p)

    for i in range(n_c1p):
        v = class1[i, :] - class2
        v2 = v ** 2
        vsum = np.sum(v2, axis=1)
        x_hat = np.min(vsum)
        lambda_1[i] = min(x_hat - eps, lambda_max)

    for i in range(n_c2p):
        v = class2[i, :] - class1
        v2 = v ** 2
        vsum = np.sum(v2, axis=1)
        x_hat = np.min(vsum)
        lambda_2[i] = min(x_hat - eps, lambda_max)

    return lambda_1, lambda_2


def rce_classify(class1, lambda_1, class2, lambda_2, test_patterns):
    # Test Patterns in form: num_features x num_patterns
    ind1 = []
    ind2 = []

    # Find number of train patterns (columns)
    n_c1p = class1.shape[0]
    n_c2p = class2.shape[0]
    num_test_patterns = test_patterns.shape[0]

    cl = np.zeros(num_test_patterns,dtype=int)

    for i in range(num_test_patterns):
        test_x = test_patterns[i,:]
        v11 = test_x - class1
        v12 = test_x - class2

        v21 = v11 ** 2
        v22 = v12 ** 2

        dist1 = np.sum(v21, axis=1)
        dist2 = np.sum(v22, axis=1)

        ind1 = np.where(dist1 < lambda_1)[0]
        ind2 = np.where(dist2 < lambda_2)[0]

        if len(ind1) != 0:
            p = 0
        elif len(ind2) != 0:
            p = 1
        else:
            p = 2

        cl[i] = p

    return cl


# Generate 100 random samples with 0.05 noise
X, y = make_moons(n_samples=200, noise=0.05, random_state=42)

# Split the data into 70% for training and 30% for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Plot the training and testing sets
# plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='viridis', label='Training set')
# plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap='viridis', marker='x', label='Testing set')
# plt.xlabel('Feature 1')
# plt.ylabel('Feature 2')
# plt.legend()
# plt.show()

class1 = X_train[y_train == 0]
class2 = X_train[y_train == 1]

eps=1e-5
lambda_max=1

l1, l2 = rce_train(class1,class2, eps, lambda_max)

y_pred = rce_classify(class1, l1, class2, l2, X_test)

# Assuming y_pred and y_test are numpy arrays or lists
cm = confusion_matrix(y_test, y_pred)

# Print the confusion matrix
print('Confusion matrix:\n', cm)

R1 = np.sqrt(l1)
C1 = class1
n_circles1 = len(R1)

R2 = np.sqrt(l2)
C2 = class2
# Assuming C and R are numpy arrays or lists
n_circles2 = len(R2)
fig, ax = plt.subplots()

for i in range(n_circles1):
    circle = plt.Circle(C1[i], R1[i], fill=False, edgecolor='red')
    ax.add_artist(circle)

for i in range(n_circles1):
    circle = plt.Circle(C2[i], R2[i], fill=False, edgecolor='blue')
    ax.add_artist(circle)

# Set the limits of the plot based on the centers and radii of the circles
ax.scatter(C1[:, 0], C1[:, 1], c='red')
ax.scatter(C2[:, 0], C2[:, 1], c='blue')
ax.set_xlim(-2.2, 3.2)
ax.set_ylim(-1.5, 2)
# Set aspect ratio to "equal" for same scale in both axes
ax.set_aspect('equal')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
# plt.legend()
# Show the plot
plt.show()