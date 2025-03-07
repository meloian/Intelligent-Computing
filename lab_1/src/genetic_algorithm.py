import numpy as np

class GA:

    def __init__(
        self, 
        f,           # objective function
        ps=50,       # population size
        bnds=[(-5, 10), (0, 15)],
        ngen=100,    # number of generations
        sel_type='tournament',
        cx_type='single_point',
        mut_type='gaussian',
        cx_p=0.8,
        mut_p=0.2,
        tour_k=3,
        sig=0.1,
        elite=True
    ):
        # storing all hyperparameters
        self.f = f
        self.ps = ps
        self.bnds = bnds
        self.ngen = ngen
        self.sel_type = sel_type
        self.cx_type = cx_type
        self.mut_type = mut_type
        self.cx_p = cx_p
        self.mut_p = mut_p
        self.tour_k = tour_k
        self.sig = sig
        self.elite = elite

        # creating initial population randomly
        self.pop = self.init_pop()
        self.hist_pop = []  # for animations 

    def init_pop(self):
        p_list = []
        for _ in range(self.ps):
            xcoord = np.random.uniform(self.bnds[0][0], self.bnds[0][1])
            ycoord = np.random.uniform(self.bnds[1][0], self.bnds[1][1])
            p_list.append([xcoord, ycoord])
        return np.array(p_list)

    def calc_fit(self, group=None):
        # fitness = -f(x,y). we do this since the GA code
        # will maximize the fitness, but we want to minimize f.
        if group is None:
            group = self.pop
        fits = []
        for ind in group:
            val = -self.f(ind[0], ind[1])
            fits.append(val)
        return np.array(fits)

    def select_parents(self, fit):
        # decide how to select: roulette or tournament
        if self.sel_type == 'roulette':
            return self.roulette_sel(fit)
        elif self.sel_type == 'tournament':
            return self.tournament_sel(fit)
        else:
            raise ValueError("Unknown selection method")

    def roulette_sel(self, fit):
        # roulette approach: shift all fitness to be positive 
        # and pick with probability ~ fitness.
        minf = np.min(fit)
        shifted = fit - minf + 1e-10
        total = np.sum(shifted)
        pars = []
        for _ in range(self.ps):
            spin = np.random.uniform(0, total)
            accum = 0
            for i in range(self.ps):
                accum += shifted[i]
                if accum >= spin:
                    pars.append(self.pop[i])
                    break
        return np.array(pars)

    def tournament_sel(self, fit):
        # sample 'tour_k' individuals, pick the best among them.
        pars = []
        for _ in range(self.ps):
            idxs = np.random.randint(0, self.ps, self.tour_k)
            best_local = np.argmax(fit[idxs])
            real_idx = idxs[best_local]
            pars.append(self.pop[real_idx])
        return np.array(pars)

    def crossover(self, pars):
        # perform crossover at probability cx_p.
        offs = []
        for i in range(0, self.ps, 2):
            p1 = pars[i]
            p2 = pars[(i + 1) % self.ps]
            if np.random.rand() < self.cx_p:
                if self.cx_type == 'single_point':
                    c1, c2 = self.cx_single(p1, p2)
                elif self.cx_type == 'two_point':
                    c1, c2 = self.cx_two(p1, p2)
                else:
                    raise ValueError("Bad crossover type")
            else:
                c1, c2 = p1.copy(), p2.copy()
            offs.append(c1)
            offs.append(c2)
        return np.array(offs)

    def cx_single(self, a, b):
        # single-point crossover: for 2D, 
        # cutting after the first coordinate.
        cut = 1
        ch1 = np.concatenate([a[:cut], b[cut:]])
        ch2 = np.concatenate([b[:cut], a[cut:]])
        return ch1, ch2

    def cx_two(self, a, b):
        # two-point crossover: in 2D, acts similarly
        # but we keep it for demonstration.
        c1, c2 = 0, 1
        part1 = np.concatenate([a[:c1], b[c1:c2], a[c2:]])
        part2 = np.concatenate([b[:c1], a[c1:c2], b[c2:]])
        return part1, part2

    def mutate(self, offspr):
        # with probability mut_p, mutate each individual.
        for i in range(len(offspr)):
            if np.random.rand() < self.mut_p:
                if self.mut_type == 'gaussian':
                    offspr[i] = self.mutate_gauss(offspr[i])
                elif self.mut_type == 'random':
                    offspr[i] = self.mutate_rand(offspr[i])
                else:
                    raise ValueError("Bad mutation type")
        return offspr

    def mutate_gauss(self, pt):
        # add normal noise, then clamp. 
        newpt = pt.copy()
        for d in range(len(newpt)):
            newpt[d] += np.random.normal(0, self.sig)
            if newpt[d] < self.bnds[d][0]:
                newpt[d] = self.bnds[d][0]
            elif newpt[d] > self.bnds[d][1]:
                newpt[d] = self.bnds[d][1]
        return newpt

    def mutate_rand(self, pt):
        # randomly reset coordinate within bounds.
        newpt = pt.copy()
        for d in range(len(newpt)):
            newpt[d] = np.random.uniform(self.bnds[d][0], self.bnds[d][1])
        return newpt

    def run(self):
        # GA loop: returns best solution, best fitness,
        # history of best fitness, and population snapshots.
        fit_vals = self.calc_fit(self.pop)
        idx_best = np.argmax(fit_vals)
        best_ind = self.pop[idx_best].copy()
        best_val = fit_vals[idx_best]
        
        hist_best = [best_val]
        self.hist_pop = [self.pop.copy()]

        for _ in range(self.ngen):
            # if elitism, keep track of best from previous gen
            if self.elite:
                saved_best = best_ind.copy()

            pars = self.select_parents(fit_vals)
            kids = self.crossover(pars)
            kids = self.mutate(kids)

            new_fit = self.calc_fit(kids)

            # reinsert best if elitism
            if self.elite:
                rand_idx = np.random.randint(0, self.ps)
                kids[rand_idx] = saved_best
                new_fit[rand_idx] = -self.f(saved_best[0], saved_best[1])

            self.pop = kids
            fit_vals = new_fit

            # update best
            idx_loc = np.argmax(fit_vals)
            val_loc = fit_vals[idx_loc]
            if val_loc > best_val:
                best_val = val_loc
                best_ind = self.pop[idx_loc].copy()

            hist_best.append(best_val)
            self.hist_pop.append(self.pop.copy())

        return best_ind, best_val, hist_best, self.hist_pop 