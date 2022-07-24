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
def fitness_(x, vehicle_num, vehicles_speed, target_num, targets, time_lim, map):
    ins = np.zeros(target_num+1, dtype=numba.int32)
    seq = np.argsort(x[vehicle_num-1:]) + 1
    gene = (x[:vehicle_num-1]*target_num).astype(numba.int32)
    ins[target_num] = 1
    for i in range(vehicle_num-1):
        ins[gene[i]] += 1
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
        self.workers = int(mp.cpu_count()/3) # leave threads for other tests
        self.retries = env.retries
        self.dim = self.target_num + self.vehicle_num - 1
       
    def name(self):
        #return self.optname.split()[0]
        return self.optname.replace(' -> ', '_').replace(' cpp', '')

    def fitness(self, x):   
        return -fitness_(x, self.vehicle_num, self.vehicles_speed, 
                           self.target_num, self.targets, self.time_lim, self.map)
        
    def run(self):
        try:
            print(self.name() + " start, pid: %s" % os.getpid())
            start_time = time.time()
            bounds = Bounds([0] * self.dim, [1] * self.dim) 
            res = retry.minimize(self.fitness, bounds, optimizer=self.opt, 
                                 num_retries=self.workers*self.retries, workers=self.workers, logger=None)
            
            x = res.x
            ins = np.zeros(self.target_num+1, dtype=int)
            seq = np.argsort(x[self.vehicle_num-1:]) + 1
            gene = (x[:self.vehicle_num-1]*self.target_num).astype(int)
            ins[self.target_num] = 1
            for i in range(self.vehicle_num-1):
                ins[gene[i]] += 1
            i = 0  # index of vehicle
            pre = 0  # index of last target
            post = 0  # index of ins/seq
            t = 0
            reward = 0
            task_assignment = [[] for i in range(self.vehicle_num)]
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
