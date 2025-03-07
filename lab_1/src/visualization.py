import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def plot_function(func, bnds, title="Function Plot", save_path=None, resolution=100):
    # plot a 3D surface of the given function
    x0, x1 = bnds[0]
    y0, y1 = bnds[1]
    xs = np.linspace(x0, x1, resolution)
    ys = np.linspace(y0, y1, resolution)
    X, Y = np.meshgrid(xs, ys)
    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i,j] = func(X[i,j], Y[i,j])

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

def plot_history(hist, title="Fitness over generations", save_path=None):
    # plot best fitness across generations
    fig, ax = plt.subplots()
    ax.plot(range(len(hist)), hist, label='Best')
    ax.set_title(title)
    ax.set_xlabel('Gen')
    ax.set_ylabel('Fitness (-f(x))')
    ax.legend()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_histories_comparison(harr, lbls, title="Comparison", save_path=None):
    # compare multiple histories on one figure
    fig, ax = plt.subplots()
    for h, label in zip(harr, lbls):
        ax.plot(h, label=label)
    ax.set_title(title)
    ax.set_xlabel("Gen")
    ax.set_ylabel("Best Fit (-f(x))")
    ax.legend()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def animate(func, bnds, hist_pop, title="animation", save_path=None):
    # 2D contour and population evolution
    x0, x1 = bnds[0]
    y0, y1 = bnds[1]
    res = 200
    xs = np.linspace(x0, x1, res)
    ys = np.linspace(y0, y1, res)
    X, Y = np.meshgrid(xs, ys)
    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i,j] = func(X[i,j], Y[i,j])

    fig, ax = plt.subplots()
    ax.contourf(X, Y, Z, levels=30, cmap='viridis')
    sc = ax.scatter([], [], color='red', s=15)
    ax.set_title(title)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)

    def init():
        sc.set_offsets([])
        return sc,

    def update(fr):
        sc.set_offsets(hist_pop[fr])
        ax.set_title(f"{title}\nGen: {fr}")
        return sc,

    anim = FuncAnimation(fig, update, frames=len(hist_pop),
                         init_func=init, blit=True, repeat=False)
    if save_path:
        anim.save(save_path, writer='pillow', fps=2)
    plt.close()

def animate_3d(func, bnds, hist_pop, title="3D animation", save_path=None):
    # 3D version: population points on function surface
    x0, x1 = bnds[0]
    y0, y1 = bnds[1]
    res = 50
    xs = np.linspace(x0, x1, res)
    ys = np.linspace(y0, y1, res)
    X, Y = np.meshgrid(xs, ys)
    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i,j] = func(X[i,j], Y[i,j])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title(title)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('f(x,y)')
    ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.7, edgecolor='none')

    sc_pop = ax.scatter([], [], [], color='red', s=10)
    sc_best = ax.scatter([], [], [], color='yellow', s=40, marker='*')

    def init():
        sc_pop._offsets3d = ([], [], [])
        sc_best._offsets3d = ([], [], [])
        return sc_pop, sc_best

    def update(fr):
        population = hist_pop[fr]
        zvals = []
        for p in population:
            zvals.append(func(p[0], p[1]))
        zvals = np.array(zvals)
        best_idx = np.argmin(zvals)
        ax.set_title(f"{title}\nGen: {fr}")

        sc_pop._offsets3d = (population[:, 0], population[:, 1], zvals)
        sc_best._offsets3d = (
            [population[best_idx, 0]],
            [population[best_idx, 1]],
            [zvals[best_idx]]
        )
        return sc_pop, sc_best

    anim = FuncAnimation(fig, update, frames=len(hist_pop),
                         init_func=init, blit=False, repeat=False)
    if save_path:
        anim.save(save_path, writer='pillow', fps=2)
    plt.close() 