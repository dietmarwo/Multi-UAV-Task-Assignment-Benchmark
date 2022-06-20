# do 'pip install fcmaes --upgrade' before executing this code

# Modifies the problem to a multi objective one. Objectives are:

# - reward (to be maximized)
# - maximal time (to be minimized)
# - energy (to be minimized)

# The maximal time constraint from the single objective case is still valid.
# Energy consumption is approximated by sum(dt*v*v) 

# Applies a standard continuous optimization MO algorithm - fcmaes MODE 
# from Dietmar Wolz - using a fitness function adapted from GA.py. 
# The fitness function uses numba for speed up. 

# Executing this file you may monitor the progress of MODE during optimization.
# On an AMD 5950 16 core processor more than 500k fitness executions per second
# are performed - half as many as for BiteOpt single objective optimization. 

# After completion single objective optimization is performed for the same 
# problem instance for comparison. For medium and large problems BiteOpt single
# objective optimization delivers a bigger award compared to MO-optimizaiton, 
# but still beats the PSO, GA and ACO results by some margin. 

# Projections of the pareto front are written to an image file, 
# together with a compressed numpy representation of the pareto fronts.
# The compressed npz pareto files can be read using:
 
#            with np.load(filename) as data:
#                xs = list(data['xs'])
#                ys = list(data['ys'])

# See https://github.com/dietmarwo/fast-cma-es/blob/master/tutorials/UAV.adoc

from scipy.optimize import Bounds
from fcmaes.optimizer import wrapper
from evaluate import Env
import numpy as np
from numba import njit
import numba
from fcmaesopt import Optimizer
from fcmaes.optimizer import Bite_cpp
from fcmaes import mode, modecpp, retry, moretry
import multiprocessing as mp

@njit(fastmath=True)
def fitness_(gene, vehicle_num, vehicles_speed, target_num, targets, time_lim, map):
    ins = np.zeros(target_num+1, dtype=numba.int32)
    seq = np.zeros(target_num, dtype=numba.int32)
    ins[target_num] = 1
    for i in range(vehicle_num-1):
        ins[gene[i]] += 1
    rest = np.zeros(target_num, dtype=numba.int32)
    for i in range(0, target_num):
        rest[i] = i+1   
    for i in range(target_num-1):
        seq[i] = rest[gene[i+vehicle_num-1]]
        rest = np.delete(rest, gene[i+vehicle_num-1])
    seq[target_num-1] = rest[0]
    i = 0  # index of vehicle
    pre = 0  # index of last target
    post = 0  # index of ins/seq
    t = 0
    reward = 0
    max_time = 0
    energy = 0
    while i < vehicle_num:
        if ins[post] > 0:
            i += 1
            ins[post] -= 1
            pre = 0
            t = 0
        else:
            t += targets[pre, 3]
            v = vehicles_speed[i]
            dt = map[pre, seq[post]]/v
            t += dt
            if t < time_lim:
                reward += targets[seq[post], 2]
                if t > max_time:
                    max_time = t
                energy += v*v*dt # approximated
            pre = seq[post]
            post += 1
    return -reward, max_time, energy # reward is to be maximized

class Fitness:

    def __init__(self, vehicle_num, vehicles_speed, target_num, targets, time_lim):
        self.dim = vehicle_num + target_num - 2
        self.vehicle_num = vehicle_num
        self.vehicles_speed = vehicles_speed
        self.target_num = target_num     
        self.targets = targets
        self.time_lim = time_lim  
        self.map = np.zeros(shape=(target_num+1, target_num+1), dtype=float)
        for i in range(target_num+1):
            self.map[i, i] = 0
            for j in range(i):
                self.map[j, i] = self.map[i, j] = np.linalg.norm(
                    targets[i, :2]-targets[j, :2])
        self.upper = np.array([target_num] * (vehicle_num-1) + list(range(target_num, 1, -1)))-1E-9
        self.bounds = Bounds([0] * self.dim, [1] * self.dim) 

    def get_gene(self, x):
        return (x*self.upper).astype(int)

    def __call__(self, x):   
        return fitness_(self.get_gene(x), self.vehicle_num, self.vehicles_speed, 
                           self.target_num, self.targets, self.time_lim, self.map)

# deliver both MO and SO problem instances for comparison
def get_fitness(vehicle_num, target_num, map_size, seed = None):
    env = Env(vehicle_num,target_num,map_size,visualized=True, seed=seed)
    return Fitness(vehicle_num,env.vehicles_speed,target_num,env.targets,env.time_lim), \
           Optimizer(env, vehicle_num,env.vehicles_speed,target_num,env.targets,env.time_lim, Bite_cpp(0))

def main():
    try:      
        nobj = 3
        evals = 5000000 # even 100000000 makes sense, but requires about 1 hour
        
        # small scale
        #mo_problem, so_problem = get_fitness(5,30,5e3)
        # medium scale
        #mo_problem, so_problem = get_fitness(10,60,1e4)
        # large scale       
        mo_problem, so_problem = get_fitness(15,90, 1.5e4, 65)
        
        mo_fun = mode.wrapper(mo_problem, nobj, interval = 1E12)
        so_fun = wrapper(so_problem.fitness)
                
        workers = mp.cpu_count()
                
        # MO parallel optimization retry
        xs, ys = modecpp.retry(mo_fun, nobj, 0, 
                      mo_problem.bounds, num_retries=workers, popsize = 512,
                  max_evaluations = evals, nsga_update = True, workers=workers)

        name = "pareto_uav"
        np.savez_compressed(name, xs=xs, ys=ys)
        moretry.plot(name, 0, xs, ys, all=False)
                        
        # SO parallel optimization retry, needs less evals than MO
        res = retry.minimize(so_fun, mo_problem.bounds, optimizer=Bite_cpp(int(evals/5)), 
                             num_retries=workers, workers=workers, logger=None)

    except Exception as ex:
        print(str(ex))  

if __name__ == '__main__':
    main()
