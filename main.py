from mpl_toolkits import mplot3d#matplotlib inline
import numpy as np
import matplotlib.pyplot as plt

def plot3DGraph():
    fig = plt.figure()
    ax = plt.axes(projection='3d')

    # Define all 3 axis
    x = np.outer(np.linspace(-3, 2, 10), np.ones(10))
    y = x.copy().T
    z = np.cos(x ** 2 + y ** 3)
    # syntax for plotting
    ax.plot_surface(x, y, z, cmap='viridis', \
                    edgecolor='green')
    ax.set_title('Surface plot geeks for geeks')
    plt.show()

if __name__ == '__main__':
    plot3DGraph()