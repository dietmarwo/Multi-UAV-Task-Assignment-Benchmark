# do 'pip install fcmaes --upgrade' before executing this code

# Applies a standard continous optimization algorithm - BiteOpt 
# from Alexey Vaneev - using the same fitness function as GA.py. 

# The fitness function uses numba for speed up. 

import numpy as np
import time
import os
from numba import njit
import numba
from fcmaes import retry
from scipy.optimize import Bounds
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
    while i < vehicle_num:
        if ins[post] > 0:
            i += 1
            ins[post] -= 1
            pre = 0
            t = 0
        else:
            t += targets[pre, 3]
            past = map[pre, seq[post]]/vehicles_speed[i]
            t += past
            if t < time_lim:
                reward += targets[seq[post], 2]
            pre = seq[post]
            post += 1
    return reward

class Optimizer():
    def __init__(self, env, vehicle_num, vehicles_speed, target_num, targets, time_lim, opt):
        self.env = env
        self.optname = opt.name
        self.opt = opt
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
        self.workers = int(mp.cpu_count()/4) # leave threads for other tests
        self.retries = env.retries
        self.upper = np.array([self.target_num] * (self.vehicle_num-1) + list(range(self.target_num, 1, -1)))-1E-9
        self.dim = len(self.upper)
       
    def name(self):
        return self.optname.split()[0]
    
    def get_gene(self, x):
        return (x*self.upper).astype(int)

    def fitness(self, x):   
        return -fitness_(self.get_gene(x), self.vehicle_num, self.vehicles_speed, 
                           self.target_num, self.targets, self.time_lim, self.map)
        
    def run(self):
        try:
            print(self.name() + " start, pid: %s" % os.getpid())
            start_time = time.time()
            bounds = Bounds([0] * self.dim, [1] * self.dim) 
            res = retry.minimize(self.fitness, bounds, optimizer=self.opt, 
                                 num_retries=self.workers*self.retries, workers=self.workers, logger=None)
            gene = self.get_gene(res.x)
    
            ins = np.zeros(self.target_num+1, dtype=np.int32)
            seq = np.zeros(self.target_num, dtype=np.int32)
            ins[self.target_num] = 1
            for i in range(self.vehicle_num-1):
                ins[gene[i]] += 1
            rest = np.array(range(1, self.target_num+1))
            for i in range(self.target_num-1):
                seq[i] = rest[gene[i+self.vehicle_num-1]]
                rest = np.delete(rest, gene[i+self.vehicle_num-1])
            seq[self.target_num-1] = rest[0]
            task_assignment = [[] for i in range(self.vehicle_num)]
            i = 0  # index of vehicle
            pre = 0  # index of last target
            post = 0  # index of ins/seq
            t = 0
            reward = 0
            while i < self.vehicle_num:
                if ins[post] > 0:
                    i += 1
                    ins[post] -= 1
                    pre = 0
                    t = 0
                else:
                    t += self.targets[pre, 3]
                    past = self.map[pre, seq[post]]/self.vehicles_speed[i]
                    t += past
                    if t < self.time_lim:
                        task_assignment[i].append(seq[post])
                        reward += self.targets[seq[post], 2]
                    pre = seq[post]
                    post += 1
            print(self.name() + " result:", reward, task_assignment)
            end_time = time.time()
            print(self.name() + " time:", end_time - start_time)
            return task_assignment, end_time - start_time
        except Exception as ex:
            print(str(ex))
