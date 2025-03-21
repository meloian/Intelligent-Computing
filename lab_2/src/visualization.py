import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def plot_function(func, bnds, title="Function Plot", save_path=None, resolution=100):
    # plot 3d surface for a 2d function
    x0, x1 = bnds[0]
    y0, y1 = bnds[1]
    xs = np.linspace(x0, x1, resolution)
    ys = np.linspace(y0, y1, resolution)
    X, Y = np.meshgrid(xs, ys)
    Z = np.zeros_like(X)

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = func([X[i, j], Y[i, j]])

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')
    ax.set_title(title)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('f(x,y)')

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_history(history, title="best over iterations", save_path=None):
    # plot best value over iterations
    plt.figure()
    plt.plot(range(len(history)), history, label='best')
    plt.title(title)
    plt.xlabel('iteration')
    plt.ylabel('best f(x)')
    plt.legend()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_histories_comparison(hist_arr, labels, title="comparison", save_path=None):
    # compare multiple runs on one plot
    plt.figure()
    for h, lbl in zip(hist_arr, labels):
        plt.plot(h, label=lbl)
    plt.title(title)
    plt.xlabel('iteration')
    plt.ylabel('best f(x)')
    plt.legend()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def animate(func, bnds, history_pop, title="population evolution", save_path=None):
    # 2d animation of swarm positions on contour
    x0, x1 = bnds[0]
    y0, y1 = bnds[1]
    res = 200
    xs = np.linspace(x0, x1, res)
    ys = np.linspace(y0, y1, res)
    X, Y = np.meshgrid(xs, ys)
    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = func([X[i, j], Y[i, j]])

    fig, ax = plt.subplots()
    ax.contourf(X, Y, Z, levels=30, cmap='viridis')
    sc = ax.scatter([], [], color='red', s=15)
    ax.set_title(title)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)

    def init():
        sc.set_offsets(np.empty((0, 2)))
        return sc,

    def update(frame):
        coords = np.array(history_pop[frame])
        if coords.ndim == 1:
            coords = coords[np.newaxis, :]
        sc.set_offsets(coords)
        ax.set_title(f"{title}\niteration: {frame}")
        return sc,

    anim = FuncAnimation(fig, update, frames=len(history_pop),
                         init_func=init, blit=True, repeat=False)

    if save_path:
        anim.save(save_path, writer='pillow', fps=2)
    plt.close()

def animate_3d(func, bnds, history_pop, title="population 3d evolution", save_path=None):
    # 3d animation of swarm on surface
    from mpl_toolkits.mplot3d import Axes3D
    x0, x1 = bnds[0]
    y0, y1 = bnds[1]
    res = 50
    xs = np.linspace(x0, x1, res)
    ys = np.linspace(y0, y1, res)
    X, Y = np.meshgrid(xs, ys)
    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = func([X[i, j], Y[i, j]])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title(title)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('f(x,y)')
    ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.6, edgecolor='none')

    sc_pop = ax.scatter([], [], [], color='red', s=10)
    sc_best = ax.scatter([], [], [], color='yellow', s=40, marker='*')

    def init():
        sc_pop._offsets3d = ([], [], [])
        sc_best._offsets3d = ([], [], [])
        return sc_pop, sc_best

    def update(frame):
        pop = np.array(history_pop[frame])
        if pop.ndim == 1:
            pop = pop[np.newaxis, :]
        zs = [func(p) for p in pop]
        sc_pop._offsets3d = (pop[:, 0], pop[:, 1], zs)
        idx_best = np.argmin(zs)
        sc_best._offsets3d = ([pop[idx_best, 0]], [pop[idx_best, 1]], [zs[idx_best]])
        ax.set_title(f"{title}\niteration: {frame}")
        return sc_pop, sc_best

    anim = FuncAnimation(fig, update, frames=len(history_pop),
                         init_func=init, blit=False, repeat=False)

    if save_path:
        anim.save(save_path, writer='pillow', fps=2)
    plt.close() 