import numpy as np

class PSO:
    def __init__(
        self,
        func,               # target function (minimize)
        swarm_size=30,
        w=0.7,              # inertia
        c1=1.4,             # cognitive component
        c2=1.4,             # social component
        bounds=[(-5, 5), (-5, 5)],
        max_iter=50,
        elite=True
    ):
        # store hyperparams
        self.func = func
        self.swarm_size = swarm_size
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.bounds = bounds
        self.dim = len(bounds)
        self.max_iter = max_iter
        self.elite = elite

        # track best values and populations
        self.history_best_vals = []
        self.history_pop = []

        # initialize swarm
        self.initialize_swarm()

    def initialize_swarm(self):
        # create positions and velocities
        self.positions = np.zeros((self.swarm_size, self.dim))
        self.velocities = np.zeros((self.swarm_size, self.dim))
        for i in range(self.swarm_size):
            for d in range(self.dim):
                low, high = self.bounds[d]
                self.positions[i, d] = np.random.uniform(low, high)
                self.velocities[i, d] = np.random.uniform(-1, 1)

        # personal and global best
        self.personal_bests = self.positions.copy()
        self.personal_best_vals = np.full(self.swarm_size, np.inf)
        self.global_best = None
        self.global_best_val = np.inf

    def evaluate_swarm(self):
        # evaluate each particle, update personal/global best
        for i in range(self.swarm_size):
            val = self.func(self.positions[i])
            if val < self.personal_best_vals[i]:
                self.personal_best_vals[i] = val
                self.personal_bests[i] = self.positions[i].copy()
            if val < self.global_best_val:
                self.global_best_val = val
                self.global_best = self.positions[i].copy()

    def iteration_step(self):
        # update velocities
        for i in range(self.swarm_size):
            r1 = np.random.rand(self.dim)
            r2 = np.random.rand(self.dim)
            cognitive = self.c1 * r1 * (self.personal_bests[i] - self.positions[i])
            social = self.c2 * r2 * (self.global_best - self.positions[i])
            self.velocities[i] = self.w * self.velocities[i] + cognitive + social

        # update positions and clamp
        for i in range(self.swarm_size):
            self.positions[i] += self.velocities[i]
            for d in range(self.dim):
                low, high = self.bounds[d]
                if self.positions[i, d] < low:
                    self.positions[i, d] = low
                elif self.positions[i, d] > high:
                    self.positions[i, d] = high

        # reevaluate swarm
        self.evaluate_swarm()

    def run(self):
        # initial evaluation
        self.evaluate_swarm()
        self.history_best_vals = [self.global_best_val]
        self.history_pop = [self.positions.copy()]

        # main loop
        for _ in range(self.max_iter):
            self.iteration_step()
            self.history_best_vals.append(self.global_best_val)
            self.history_pop.append(self.positions.copy())

        return self.global_best, self.global_best_val, self.history_best_vals, self.history_pop 