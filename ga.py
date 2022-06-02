import numpy as np
import random
import time
import os
from numba import njit
from numba.typed import List
import numba

# Performance is largely improved by using https://numba.pydata.org/
# See https://github.com/dietmarwo/fast-cma-es/blob/master/tutorials/UAV.adoc

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

@njit(fastmath=True)
def selection_(tmp_ff, ff, pop_size, tmp_size, pop, tmp_pop):
    roll = np.zeros(tmp_size)
    roll[0] = tmp_ff[0]
    for i in range(1, tmp_size):
        roll[i] = roll[i-1]+tmp_ff[i]
    for i in range(pop_size):
        xx = random.uniform(0, roll[tmp_size-1])
        j = 0
        while xx > roll[j]:
            j += 1
        pop[i, :] = tmp_pop[j, :]
        ff[i] = tmp_ff[j]

@njit(fastmath=True)
def mutation_(tmp_ff, p_mutate, tmp_size, tmp_pop, pop, vehicle_num, vehicles_speed, target_num, targets, time_lim, map):
    for i in range(tmp_size):
        flag = False
        for j in range(vehicle_num-1):
            if random.random() < p_mutate:
                tmp_pop[i, j] = random.randint(0, target_num)
                flag = True
        for j in range(target_num-1):
            if random.random() < p_mutate:
                tmp_pop[i, vehicle_num+j -
                             1] = random.randint(0, target_num-j-1)
                flag = True
        if flag:
            tmp_ff[i] = fitness_(tmp_pop[i, :], vehicle_num, vehicles_speed, target_num, targets, time_lim, map)

@njit(fastmath=True)
def crossover_(ff, p_cross, pop_size, pop, vehicle_num, vehicles_speed, target_num, targets, time_lim, map):
    new_pop = List()
    new_ff = List()
    new_size = 0
    for i in range(0, pop_size, 2):
        if random.random() < p_cross:
            x1 = random.randint(0, vehicle_num-2)
            x2 = random.randint(0, target_num-2)+vehicle_num
            g1 = pop[i, :]
            g2 = pop[i+1, :]
            g1[x1:x2] = pop[i+1, x1:x2]
            g2[x1:x2] = pop[i, x1:x2]
            new_pop.append(g1)
            new_pop.append(g2)
            new_ff.append(fitness_(g1, vehicle_num, vehicles_speed, target_num, targets, time_lim, map))
            new_ff.append(fitness_(g2, vehicle_num, vehicles_speed, target_num, targets, time_lim, map))
            new_size += 2
    tmp_size = pop_size+new_size
    tmp_pop = np.zeros(
        shape=(tmp_size, vehicle_num-1+target_num-1), dtype=numba.int32)
    tmp_pop[0:pop_size, :] = pop
    tmp_ff = np.zeros(tmp_size)
    tmp_ff[0:pop_size] = ff
    for i in range(pop_size, tmp_size):
        tmp_pop[i,:] = new_pop[i-pop_size]
        tmp_ff[i] = new_ff[i-pop_size]
    return tmp_pop, tmp_ff, tmp_size

class GA():
    def __init__(self, vehicle_num, vehicles_speed, target_num, targets, time_lim):
        # vehicles_speed,targets in the type of narray
        self.vehicle_num = vehicle_num
        self.vehicles_speed = vehicles_speed
        self.target_num = target_num
        self.targets = targets
        self.time_lim = time_lim
        self.map = np.zeros(shape=(target_num+1, target_num+1), dtype=float)
        self.pop_size = 300
        self.p_cross = 0.6
        self.p_mutate = 0.005
        for i in range(target_num+1):
            self.map[i, i] = 0
            for j in range(i):
                self.map[j, i] = self.map[i, j] = np.linalg.norm(
                    targets[i, :2]-targets[j, :2])
        self.pop = np.zeros(
            shape=(self.pop_size, vehicle_num-1+target_num-1), dtype=int)
        self.ff = np.zeros(self.pop_size, dtype=float)
        for i in range(self.pop_size):
            for j in range(vehicle_num-1):
                self.pop[i, j] = random.randint(0, target_num)
            for j in range(target_num-1):
                self.pop[i, vehicle_num+j -
                         1] = random.randint(0, target_num-j-1)
            self.ff[i] = self.fitness(self.pop[i, :])
        self.tmp_pop = None
        self.tmp_ff = None
        self.tmp_size = 0
        
    def name(self):
        return "GA"

    def fitness(self, gene):   
        return fitness_(gene, self.vehicle_num, self.vehicles_speed, 
                           self.target_num, self.targets, self.time_lim, self.map)
    
    def selection(self):
        selection_(self.tmp_ff, self.ff, self.pop_size, self.tmp_size, self.pop, self.tmp_pop)

    def mutation(self):
        mutation_(self.tmp_ff, self.p_mutate, self.tmp_size, self.tmp_pop, self.pop, 
                  self.vehicle_num, self.vehicles_speed, self.target_num, self.targets, self.time_lim, self.map)
        
    def crossover(self):
        self.tmp_pop, self.tmp_ff, self.tmp_size = crossover_(
                    self.ff, self.p_cross, self.pop_size, self.pop, 
                    self.vehicle_num, self.vehicles_speed, self.target_num, self.targets, self.time_lim, self.map)

    def run(self):
        print("GA start, pid: %s" % os.getpid())
        start_time = time.time()
        cut = 0
        count = 0
        while count < 6000:
            self.crossover()
            self.mutation()
            self.selection()
            new_cut = self.tmp_ff.max()
            if cut < new_cut:
                cut = new_cut
                count = 0
                gene = self.tmp_pop[np.argmax(self.tmp_ff)]
            else:
                count += 1

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
        print("GA result:", reward, task_assignment)
        end_time = time.time()
        print("GA time:", end_time - start_time)
        return task_assignment, end_time - start_time

