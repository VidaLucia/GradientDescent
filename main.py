import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from autograd import grad

# Our function we want to minimize
def costFunction(v): #v contains [x,y]
    x,y = v[0],v[1]
    return x**2-x/3 + x* y**2

# Gradient of the function
#def gradient(x, y):
    #return 2*x, 2*y

gradient = grad(costFunction)
# Run gradient descent and return the path of points
def descentPathFunc(lr=0.1, steps=100, start=(4.0, 4.0)):
    path = []
    point = np.array(start, dtype=np.float64)
    for _ in range(steps):
        path.append((point[0], point[1], costFunction(point)))
        point -= lr * gradient(point)
    return np.array(path)

def plot3DGraph():
    # Create the 3D surface grid
    x = np.arange(-4, 4, 0.05)
    y = np.arange(-4, 4, 0.05)
    X, Y = np.meshgrid(x, y)
    Z = np.array([[costFunction(np.array([xi, yi])) for xi, yi in zip(row_x, row_y)] for row_x, row_y in zip(X, Y)])

    # Get descent path
    path = descentPathFunc(lr=0.01, steps=200, start=(3, 3))
    xs, ys, zs = path[:, 0], path[:, 1], path[:, 2]

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    # Plot surface once
    ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.6, zorder=0)

    # Initial scatter point (magenta)
    point_plot = ax.scatter([xs[0]], [ys[0]], [zs[0]], color='magenta', s=50, zorder=1)

    # Line showing path so far
    path_line, = ax.plot([], [], [], 'm--', linewidth=1.5)

    # Set static plot settings
    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 4)
    ax.set_zlim(0, 35)
    ax.set_title('Gradient Descent on 3D Surface')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('f(x, y)')

    def update(i):
        point_plot._offsets3d = ([xs[i]], [ys[i]], [zs[i]])
        path_line.set_data(xs[:i+1], ys[:i+1])
        path_line.set_3d_properties(zs[:i+1])
        return point_plot, path_line

    ani = animation.FuncAnimation(fig, update, frames=len(xs), interval=30, blit=False)
    plt.show()

if __name__ == '__main__':
    plot3DGraph()
