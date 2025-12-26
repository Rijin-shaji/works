import numpy as np
import matplotlib.pyplot as plt

def plot_svm_decision_boundary(model, X, y):
    # Convert to numpy
    X = np.array(X)
    y = np.array(y)

    # Create grid
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 500),
        np.linspace(y_min, y_max, 500)
    )

    # Predict grid points
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Plot contour (decision boundary)
    plt.contourf(xx, yy, Z, alpha=0.3)
    
    # Plot original points
    plt.scatter(X[:,0], X[:,1], c=y, s=30, edgecolors='k')

    plt.title("SVM Decision Boundary")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()
