from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Our function we want to minimize
def costFunction(x, y):
    return x**2 + y**2

# The gradient (partial derivatives) of the function
#returns x derivative and y derivative
def gradient(x, y):
    return 2*x, 2*y

# Run gradient descent and return the path of points
def descentPathFunc(lr=0.1, steps=50, start=(4.0, 4.0)):
    path = []
    point = np.array(start)
    for _ in range(steps):
        path.append((point[0], point[1], costFunction(*point)))
        point -= lr * gradient(*point)
    return np.array(path)

def plot3DGraph():
    x = np.arange(-4,4,0.05)
    y = np.arange(-4,4,0.05)

    X,Y = np.meshgrid(x,y)
    Z = costFunction(X,Y)

    currentPos = (3,3,costFunction(0.4,0.4))
    #learning rate
    lr = 0.01
    steps = 1000

    ax = plt.subplot(projection='3d',computed_zorder=False)
    for _ in range(steps):
        xDerivative,yDerivative = gradient(currentPos[0],currentPos[1])
        XNew,YNew = currentPos[0]-lr * xDerivative,currentPos[1]-lr*yDerivative
        currentPos = (XNew,YNew,costFunction(XNew,YNew))

        ax.plot_surface(X,Y,Z, cmap ="viridis",zorder=0)
        ax.scatter(currentPos[0], currentPos[1], currentPos[2],color ="magenta",zorder=1)
        plt.pause(0.001)
        ax.clear()


if __name__ == '__main__':
    plot3DGraph()
