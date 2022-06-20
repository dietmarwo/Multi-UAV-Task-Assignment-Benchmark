# do 'pip install fcmaes --upgrade' before executing this code

# Changes to the original code:

# 1) GA uses numba for a dramatic speedup. Parameters are adapted so that the
#     execution time remains the same: popsize 50 -> 300, iterations 500 -> 6000
#     For this reason GA performs much better than the original

# 2) Experiments are configured so that wall time for small size is balanced. This means
#     - increased effort for GA
#     - decreased effort for ACO. 

# 3) Adds a standard continuous optimization algorithms
#    BiteOpt, CR-FM-NES and DE using the same fitness function as GA.py. 

# 4) Uses NestablePool to enable BiteOpt multiprocessing - many optimization runs
#    are performed in parallel and the best result is returned. 

import numpy as np
import matplotlib.pyplot as plt
import random
import pandas as pd
import copy
import multiprocessing.pool
from ga import GA
from aco import ACO
from pso import PSO
from fcmaesopt import Optimizer
from fcmaes.optimizer import Bite_cpp, Cma_cpp, Crfmnes_cpp

import multiprocessing as mp
import seaborn as sns

class NoDaemonProcess(multiprocessing.Process):
    @property
    def daemon(self):
        return False

    @daemon.setter
    def daemon(self, value):
        pass

class NoDaemonContext(type(multiprocessing.get_context())):
    Process = NoDaemonProcess

class NestablePool(multiprocessing.pool.Pool):
    def __init__(self, *args, **kwargs):
        kwargs['context'] = NoDaemonContext()
        super(NestablePool, self).__init__(*args, **kwargs)

class Env():
    def __init__(self, vehicle_num, target_num, map_size, visualized=True, 
                 time_cost=None, repeat_cost=None, seed = None):
        if not seed is None:
            random.seed(seed)
        self.vehicles_position = np.zeros(vehicle_num,dtype=np.int32)
        self.vehicles_speed = np.zeros(vehicle_num,dtype=np.int32)
        self.targets = np.zeros(shape=(target_num+1,4),dtype=np.int32)
        if vehicle_num==5:
            self.size='small'
            self.evals = 1000000
            self.retries = 1
        if vehicle_num==10:
            self.size='medium'
            self.evals = 1500000
            self.retries = 1
        if vehicle_num==15:
            self.size='large'
            self.evals = 2000000
            self.retries = 1
        self.map_size = map_size
        self.speed_range = [10, 15, 30]
        #self.time_lim = 1e6
        self.time_lim = self.map_size / self.speed_range[1]
        self.vehicles_lefttime = np.ones(vehicle_num,dtype=np.float32) * self.time_lim
        self.distant_mat = np.zeros((target_num+1,target_num+1),dtype=np.float32)
        self.total_reward = 0
        self.reward = 0
        self.visualized = visualized
        self.time = 0
        self.time_cost = time_cost
        self.repeat_cost = repeat_cost
        self.end = False
        self.assignment = [[] for i in range(vehicle_num)]
        self.task_generator()
        
    def task_generator(self):
        for i in range(self.vehicles_speed.shape[0]):
            choose = random.randint(0,2)
            self.vehicles_speed[i] = self.speed_range[choose]
        for i in range(self.targets.shape[0]-1):
            self.targets[i+1,0] = random.randint(1,self.map_size) - 0.5*self.map_size # x position
            self.targets[i+1,1] = random.randint(1,self.map_size) - 0.5*self.map_size # y position
            self.targets[i+1,2] = random.randint(1,10) # reward
            self.targets[i+1,3] = random.randint(5,30) # time consumption to finish the mission  
        for i in range(self.targets.shape[0]):
            for j in range(self.targets.shape[0]):
                self.distant_mat[i,j] = np.linalg.norm(self.targets[i,:2]-self.targets[j,:2])
        self.targets_value = copy.deepcopy((self.targets[:,2]))
        
    def step(self, action):
        count = 0
        for j in range(len(action)):
            k = action[j]
            delta_time = self.distant_mat[self.vehicles_position[j],k] / self.vehicles_speed[j] + self.targets[k,3]
            self.vehicles_lefttime[j] = self.vehicles_lefttime[j] - delta_time
            if self.vehicles_lefttime[j] < 0:
                count = count + 1
                continue
            else:
                if k == 0:
                    self.reward = - self.repeat_cost
                else:
                    self.reward = self.targets[k,2] - delta_time * self.time_cost + self.targets[k,2]
                    if self.targets[k,2] == 0:
                        self.reward = self.reward - self.repeat_cost
                    self.vehicles_position[j] = k
                    self.targets[k,2] = 0
                self.total_reward = self.total_reward + self.reward
            self.assignment[j].append(action)
        if count == len(action):
            self.end = True
        
    def run(self, assignment, algorithm, play, rond):
        self.assignment = assignment
        self.algorithm = algorithm
        self.play = play
        self.rond = rond
        self.get_total_reward()
        if self.visualized:
            self.visualize()        
            
    def reset(self):
        self.vehicles_position = np.zeros(self.vehicles_position.shape[0],dtype=np.int32)
        self.vehicles_lefttime = np.ones(self.vehicles_position.shape[0],dtype=np.float32) * self.time_lim
        self.targets[:,2] = self.targets_value
        self.total_reward = 0
        self.reward = 0
        self.end = False
        
    def get_total_reward(self):
        for i in range(len(self.assignment)):
            speed = self.vehicles_speed[i]
            for j in range(len(self.assignment[i])):
                position = self.targets[self.assignment[i][j],:4]
                self.total_reward = self.total_reward + position[2]
                if j == 0:
                    self.vehicles_lefttime[i] = self.vehicles_lefttime[i] - np.linalg.norm(position[:2]) / speed - position[3]
                else:
                    self.vehicles_lefttime[i] = self.vehicles_lefttime[i] - np.linalg.norm(position[:2]-position_last[:2]) / speed - position[3]
                position_last = position
                if self.vehicles_lefttime[i] > self.time_lim:
                    self.end = True
                    break
            if self.end:
                self.total_reward = 0
                break
            
    def visualize(self):
        if self.assignment == None:
            plt.scatter(x=0,y=0,s=200,c='k')
            plt.scatter(x=self.targets[1:,0],y=self.targets[1:,1],s=self.targets[1:,2]*10,c='r')
            plt.title('Target distribution')
            plt.savefig('task_pic/'+self.size+'/'+self.algorithm+ "-%d-%d.png" % (self.play,self.rond))
            plt.cla()
        else:
            plt.title('Task assignment by '+self.algorithm +', total reward : '+str(self.total_reward))     
            plt.scatter(x=0,y=0,s=200,c='k')
            plt.scatter(x=self.targets[1:,0],y=self.targets[1:,1],s=self.targets[1:,2]*10,c='r')
            for i in range(len(self.assignment)):
                trajectory = np.array([[0,0,20]])
                for j in range(len(self.assignment[i])):
                    position = self.targets[self.assignment[i][j],:3]
                    trajectory = np.insert(trajectory,j+1,values=position,axis=0)  
                plt.scatter(x=trajectory[1:,0],y=trajectory[1:,1],s=trajectory[1:,2]*10,c='b')
                plt.plot(trajectory[:,0], trajectory[:,1]) 
            plt.savefig('task_pic/'+self.size+'/'+self.algorithm+ "-%d-%d.png" % (self.play,self.rond))
            plt.cla()
            
def evaluate(vehicle_num, target_num, map_size):
    if vehicle_num==5:
        size='small'
    if vehicle_num==10:
        size='medium'
    if vehicle_num==15:
        size='large'
    num = 5
    onum = 6
    re_opt = []
    for _ in range(onum):       
        re_opt.append([[] for i in range(num)])
    for i in range(num):
        env = Env(vehicle_num,target_num,map_size,visualized=True,seed=37*i+13)
        for j in range(num):
            opt_result = []
            p=NestablePool(mp.cpu_count())
            opt = [GA(vehicle_num,env.vehicles_speed,target_num,env.targets,env.time_lim),
                   ACO(vehicle_num,target_num,env.vehicles_speed,env.targets,env.time_lim),
                   PSO(vehicle_num,target_num ,env.targets,env.vehicles_speed,env.time_lim),
                   Optimizer(env,vehicle_num,env.vehicles_speed,target_num,env.targets,env.time_lim, Bite_cpp(env.evals)),
                   Optimizer(env,vehicle_num,env.vehicles_speed,target_num,env.targets,env.time_lim, Crfmnes_cpp(env.evals, popsize=128)),
                   # we have to disable premature termination of CMA-ES
                   Optimizer(env,vehicle_num,env.vehicles_speed,target_num,env.targets,env.time_lim, Cma_cpp(env.evals, popsize=128, stop_hist=0))]
            for k in range(onum):       
                opt_result.append(p.apply_async(opt[k].run))
            p.close()
            p.join()
            for k in range(onum): 
                opt_task_assignment = opt_result[k].get()[0]
                env.run(opt_task_assignment,opt[k].name(),i+1,j+1)
                re_opt[k][i].append((env.total_reward,opt_result[k].get()[1]))
                env.reset()

    x_index=np.arange(num)
    ymax1 = [[] for i in range(onum)]
    ymax2 = [[] for i in range(onum)]
    ymean1 = [[] for i in range(onum)]
    ymean2 = [[] for i in range(onum)]
  
    for i in range(num):
        for k in range(onum): 
            tmp1=[re_opt[k][i][j][0] for j in range(num)]
            tmp2=[re_opt[k][i][j][1] for j in range(num)]
            ymax1[k].append(np.amax(tmp1))
            ymax2[k].append(np.amax(tmp2))
            ymean1[k].append(np.mean(tmp1))
            ymean2[k].append(np.mean(tmp2))
 
    rects = []
    cols = sns.color_palette()
    for k in range(onum): 
        rects.append(plt.bar(x_index + 0.1*k, ymax1[k],width=0.1,color=cols[k],label=opt[k].name() + '_max_reward'))
    plt.xticks(x_index+0.1,x_index)
    plt.legend()
    plt.title('max_reward_for_'+size+'_size')
    plt.savefig('max_reward_'+size+'.png')
    plt.cla()
    for k in range(onum): 
        rects.append(plt.bar(x_index + 0.1*k, ymax2[k],width=0.1,color=cols[k],label=opt[k].name() + '_max_time'))
    plt.xticks(x_index+0.1,x_index)
    plt.legend()
    plt.title('max_time_for_'+size+'_size')
    plt.savefig('max_time_'+size+'.png')
    plt.cla()

    for k in range(onum): 
        rects.append(plt.bar(x_index + 0.1*k, ymean1[k],width=0.1,color=cols[k],label=opt[k].name() + '_mean_reward'))
    plt.xticks(x_index+0.1,x_index)
    plt.legend()
    plt.title('mean_reward_for_'+size+'_size')
    plt.savefig('mean_reward_'+size+'.png')
    plt.cla()
    
    for k in range(onum): 
        rects.append(plt.bar(x_index + 0.1*k, ymean2[k],width=0.1,color=cols[k],label=opt[k].name() + '_mean_time'))
    plt.xticks(x_index+0.1,x_index)
    plt.legend()
    plt.title('mean_time_for_'+size+'_size')
    plt.savefig('mean_time_'+size+'.png')
    plt.cla()
     
    t_opt = [[] for i in range(onum)]
    r_opt = [[] for i in range(onum)]

    for i in range(num):
        for j in range(num):
            for k in range(onum): 
                t_opt[k].append(re_opt[k][i][j][1])
                r_opt[k].append(re_opt[k][i][j][0])
    optdict = {}
    for k in range(onum):     
        optdict[opt[k].name() + '_time'] = t_opt[k]
        optdict[opt[k].name() + '_reward'] = r_opt[k]
    dataframe = pd.DataFrame(optdict)
    dataframe.to_csv(size+'_size_result.csv',sep=',')
    
if __name__=='__main__':
    # small scale
    evaluate(5,30,5e3)
    # # medium scale
    evaluate(10,60,1e4)
    # large scale
    evaluate(15,90,1.5e4)
