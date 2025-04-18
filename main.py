from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Our function we want to minimize
def costFunction(x, y):
    return x**2 + y**2

# The gradient (partial derivatives) of the function
def gradient(x, y):
    return np.array([2*x, 2*y])

# Run gradient descent and return the path of points
def descentPathFunc(lr=0.01, steps=50, start=(4.0, 4.0)):
    path = []
    point = np.array(start)
    for _ in range(steps):
        path.append((point[0], point[1], costFunction(*point)))
        point -= lr * gradient(*point)
    return np.array(path)

def plot3DGraph():
    fig = plt.figure()
    ax = plt.axes(projection='3d')

    # Define 3D surface
    x = np.linspace(-5, 5, 50)
    y = np.linspace(-5, 5, 50)
    X, Y = np.meshgrid(x, y)
    Z = costFunction(X, Y)

    # Compute gradient descent path
    path = descentPathFunc()
    xs, ys, zs = path[:, 0], path[:, 1], path[:, 2]

    # Plot surface and initial point
    ax.plot_surface(X, Y, Z, alpha=0.2, cmap='viridis', edgecolor='none')
    descentPoint, = ax.plot([xs[0]], [ys[0]], [zs[0]], 'ro')  # point
    descentPath, = ax.plot([], [], [], 'r--')  # trail

    # Animation function
    def update(i):
        descentPoint.set_data([xs[i]], [ys[i]])
        descentPoint.set_3d_properties([zs[i]])
        descentPath.set_data(xs[:i + 1], ys[:i + 1])
        descentPath.set_3d_properties(zs[:i + 1])
        return descentPoint, descentPath

    ani = animation.FuncAnimation(fig, update, frames=len(xs), interval=200, blit=False)

    ax.set_title('Gradient Descent on 3D Surface')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('f(x, y)')
    plt.show()

if __name__ == '__main__':
    plot3DGraph()
