# optimizers.py
import numpy as np

class SimplexMethod:
    def __init__(self, cost_func, x0, step=0.5, tol=1e-5, max_iter=50):
        self.func = cost_func
        self.tol, self.max_iter, self.n = tol, max_iter, len(x0)
        self.simplex = np.zeros((self.n + 1, self.n))
        self.simplex[0] = np.array(x0)
        for i in range(self.n):
            pt = np.array(x0); pt[i] += pt[i] * step if pt[i] != 0 else 0.1
            self.simplex[i + 1] = pt
            
    def optimize(self):
        vals = np.array([self.func(x) for x in self.simplex])
        for k in range(self.max_iter):
            order = np.argsort(vals)
            self.simplex, vals = self.simplex[order], vals[order]
            if np.abs(vals[-1] - vals[0]) < self.tol: break
            centroid = np.mean(self.simplex[:-1], axis=0)
            xr = centroid + 1.0 * (centroid - self.simplex[-1])
            fr = self.func(xr)
            if vals[0] <= fr < vals[-2]:
                self.simplex[-1], vals[-1] = xr, fr
                continue
            if fr < vals[0]:
                xe = centroid + 2.0 * (xr - centroid); fe = self.func(xe)
                if fe < fr: self.simplex[-1], vals[-1] = xe, fe
                else: self.simplex[-1], vals[-1] = xr, fr
                continue
            xc = centroid + 0.5 * (self.simplex[-1] - centroid); fc = self.func(xc)
            if fc < vals[-1]: self.simplex[-1], vals[-1] = xc, fc
            else:
                for i in range(1, self.n + 1):
                    self.simplex[i] = self.simplex[0] + 0.5 * (self.simplex[i] - self.simplex[0])
                    vals[i] = self.func(self.simplex[i])
        return self.simplex[0]

class ParticleSwarmOptimization:
    def __init__(self, cost_func, bounds, num_particles=20, max_iter=30, w=0.7, c1=1.5, c2=1.5):
        self.cost_func = cost_func
        self.bounds = np.array(bounds)
        self.n_p, self.max_iter = num_particles, max_iter
        self.w, self.c1, self.c2, self.dim = w, c1, c2, len(bounds)

    def optimize(self):
        X = np.random.uniform(self.bounds[:,0], self.bounds[:,1], (self.n_p, self.dim))
        V = np.random.uniform(-1, 1, (self.n_p, self.dim))
        P_best = X.copy()
        P_best_scores = np.array([self.cost_func(x) for x in X])
        g_best = P_best[np.argmin(P_best_scores)]
        g_best_score = np.min(P_best_scores)
        
        for i in range(self.max_iter):
            r1, r2 = np.random.rand(self.n_p, self.dim), np.random.rand(self.n_p, self.dim)
            V = self.w * V + self.c1 * r1 * (P_best - X) + self.c2 * r2 * (g_best - X)
            X = np.clip(X + V, self.bounds[:,0], self.bounds[:,1])
            scores = np.array([self.cost_func(x) for x in X])
            better = scores < P_best_scores
            P_best[better], P_best_scores[better] = X[better], scores[better]
            if np.min(scores) < g_best_score:
                g_best, g_best_score = X[np.argmin(scores)], np.min(scores)
            print(f"PSO Iter {i}: Cost {g_best_score:.4f}", end='\r')
        print("")
        return g_best

class GeneticAlgorithm:
    def __init__(self, cost_func, bounds, pop_size=20, generations=30, mut_rate=0.1):
        self.cost_func, self.bounds = cost_func, np.array(bounds)
        self.pop_size, self.gens, self.mut_rate, self.dim = pop_size, generations, mut_rate, len(bounds)

    def optimize(self):
        pop = np.random.uniform(self.bounds[:,0], self.bounds[:,1], (self.pop_size, self.dim))
        for g in range(self.gens):
            scores = np.array([self.cost_func(ind) for ind in pop])
            best_ind = pop[np.argmin(scores)]
            print(f"GA Gen {g}: Cost {np.min(scores):.4f}", end='\r')
            new_pop = [best_ind]
            while len(new_pop) < self.pop_size:
                p1, p2 = pop[np.random.randint(self.pop_size)], pop[np.random.randint(self.pop_size)]
                child = np.clip(0.5*p1 + 0.5*p2 + (np.random.normal(0, 1, self.dim) if np.random.rand() < self.mut_rate else 0), self.bounds[:,0], self.bounds[:,1])
                new_pop.append(child)
            pop = np.array(new_pop)
        print("")
        return best_ind