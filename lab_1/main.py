import os
import numpy as np

from src.genetic_algorithm import GA
from src.fitness_functions import branin, easom, goldstein_price, six_hump_camel
from src.visualization import (
    plot_function,
    plot_history,
    plot_histories_comparison,
    animate,
    animate_3d
)
import matplotlib.pyplot as plt

def main():
    np.random.seed(42)
    os.makedirs("results", exist_ok=True)

    # ----------------- Branin -----------------
    print("=== Branin ===")
    plot_function(branin, [(-5, 10), (0, 15)], "Branin", "results/branin_3d.png")

    ga_branin_1 = GA(
        f=branin,
        ps=20,
        bnds=[(-5, 10), (0, 15)],
        ngen=25,
        sel_type='tournament',
        cx_type='single_point',
        mut_type='gaussian',
        cx_p=0.6,
        mut_p=0.45,
        tour_k=2,
        sig=0.5,
        elite=True
    )
    b1best, b1fit, b1hist, b1pop = ga_branin_1.run()
    print("Branin Run1: best =", b1best, "; f(x) =", -b1fit)
    plot_history(b1hist, "Branin Run1", "results/branin_run1_fitness.png")

    ga_branin_2 = GA(
        f=branin,
        ps=50,
        bnds=[(-5, 10), (0, 15)],
        ngen=60,
        sel_type='tournament',
        cx_type='two_point',
        mut_type='gaussian',
        cx_p=0.85,
        mut_p=0.25,
        tour_k=3,
        sig=0.25,
        elite=True
    )
    b2best, b2fit, b2hist, b2pop = ga_branin_2.run()
    print("Branin Run2: best =", b2best, "; f(x) =", -b2fit)
    plot_history(b2hist, "Branin Run2", "results/branin_run2_fitness.png")

    plot_histories_comparison([b1hist, b2hist], ["Branin Run1", "Branin Run2"],
                              "Branin Compare", "results/branin_compare.png")

    animate_3d(branin, ga_branin_1.bnds, b1pop,
                            "Branin Run1", "results/branin_run1_3d.gif")
    animate_3d(branin, ga_branin_2.bnds, b2pop,
                            "Branin Run2", "results/branin_run2_3d.gif")

    # ----------------- Easom -----------------
    print("\n=== Easom ===")
    easom_bnds = [(np.pi - 3, np.pi + 3), (np.pi - 3, np.pi + 3)]
    plot_function(easom, easom_bnds, "Easom", "results/easom_3d.png")

    ga_e1 = GA(
        f=easom,
        ps=30,
        bnds=easom_bnds,
        ngen=30,
        sel_type='tournament',
        cx_type='single_point',
        mut_type='gaussian',
        cx_p=0.5,
        mut_p=0.4,
        sig=0.5,
        elite=True
    )
    e1best, e1fit, e1hist, e1pop = ga_e1.run()
    print("Easom Run1: best =", e1best, "; f(x) =", -e1fit)
    plot_history(e1hist, "Easom Run1", "results/easom_run1_fitness.png")

    ga_e2 = GA(
        f=easom,
        ps=80,
        bnds=easom_bnds,
        ngen=80,
        sel_type='roulette',
        cx_type='two_point',
        mut_type='gaussian',
        cx_p=0.85,
        mut_p=0.2,
        sig=0.3,
        elite=True
    )
    e2best, e2fit, e2hist, e2pop = ga_e2.run()
    print("Easom Run2: best =", e2best, "; f(x) =", -e2fit)
    plot_history(e2hist, "Easom Run2", "results/easom_run2_fitness.png")

    plot_histories_comparison([e1hist, e2hist], ["Easom Run1", "Easom Run2"],
                              "Easom Compare", "results/easom_compare.png")
    animate_3d(easom, easom_bnds, e1pop,
                            "Easom Run1", "results/easom_run1_3d.gif")
    animate_3d(easom, easom_bnds, e2pop,
                            "Easom Run2", "results/easom_run2_3d.gif")

    # ----------------- Goldstein-Price -----------------
    print("\n=== Goldstein-Price ===")
    gp_bnds = [(-2, 2), (-2, 2)]
    plot_function(goldstein_price, gp_bnds, "Goldstein-Price", "results/goldstein_price_3d.png")

    ga_g1 = GA(
        f=goldstein_price,
        ps=20,
        bnds=gp_bnds,
        ngen=25,
        sel_type='tournament',
        cx_type='single_point',
        mut_type='random',
        cx_p=0.5,
        mut_p=0.45,
        sig=0.2,
        elite=True
    )
    g1best, g1fit, g1hist, g1pop = ga_g1.run()
    print("Goldstein-Price Run1: best =", g1best, "; f(x) =", -g1fit)
    plot_history(g1hist, "GP Run1", "results/goldstein_price_run1_fitness.png")

    ga_g2 = GA(
        f=goldstein_price,
        ps=50,
        bnds=gp_bnds,
        ngen=60,
        sel_type='roulette',
        cx_type='two_point',
        mut_type='gaussian',
        cx_p=0.85,
        mut_p=0.2,
        sig=0.1,
        elite=True
    )
    g2best, g2fit, g2hist, g2pop = ga_g2.run()
    print("Goldstein-Price Run2: best =", g2best, "; f(x) =", -g2fit)
    plot_history(g2hist, "GP Run2", "results/goldstein_price_run2_fitness.png")

    plot_histories_comparison([g1hist, g2hist], ["GP Run1", "GP Run2"],
                              "GP Compare", "results/goldstein_price_compare.png")
    animate_3d(goldstein_price, gp_bnds, g1pop,
                            "GP Run1", "results/goldstein_price_run1_3d.gif")
    animate_3d(goldstein_price, gp_bnds, g2pop,
                            "GP Run2", "results/goldstein_price_run2_3d.gif")

    # ----------------- Six-hump Camel -----------------
    print("\n=== Six-hump Camel ===")
    shc_bnds = [(-3, 3), (-2, 2)]
    plot_function(six_hump_camel, shc_bnds, "Six-hump Camel", "results/six_hump_camel_3d.png")

    ga_c1 = GA(
        f=six_hump_camel,
        ps=20,
        bnds=shc_bnds,
        ngen=25,
        sel_type='tournament',
        cx_type='single_point',
        mut_type='gaussian',
        cx_p=0.6,
        mut_p=0.45,
        sig=0.3,
        elite=True
    )
    c1best, c1fit, c1hist, c1pop = ga_c1.run()
    print("Six-hump Camel Run1: best =", c1best, "; f(x) =", -c1fit)
    plot_history(c1hist, "SHC Run1", "results/six_hump_camel_run1_fitness.png")

    ga_c2 = GA(
        f=six_hump_camel,
        ps=40,
        bnds=shc_bnds,
        ngen=50,
        sel_type='roulette',
        cx_type='two_point',
        mut_type='random',
        cx_p=0.8,
        mut_p=0.2,
        sig=0.2,
        elite=True
    )
    c2best, c2fit, c2hist, c2pop = ga_c2.run()
    print("Six-hump Camel Run2: best =", c2best, "; f(x) =", -c2fit)
    plot_history(c2hist, "SHC Run2", "results/six_hump_camel_run2_fitness.png")

    plot_histories_comparison([c1hist, c2hist], ["SHC Run1", "SHC Run2"],
                              "SHC Compare", "results/six_hump_camel_compare.png")
    animate_3d(six_hump_camel, shc_bnds, c1pop,
                            "SHC Run1", "results/six_hump_camel_run1_3d.gif")
    animate_3d(six_hump_camel, shc_bnds, c2pop,
                            "SHC Run2", "results/six_hump_camel_run2_3d.gif")

    print("\nAll experiments finished. Results in 'results/' folder.")

if __name__ == "__main__":
    main() 