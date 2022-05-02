# do 'pip install fcmaes --upgrade' before executing this code

# Applies a standard continuous optimization algorithm - BiteOpt 
# from Alexey Vaneev - using the same fitness function as GA.py. 
# The fitness function uses numba for speed up. 

# Executing this file you may monitor the progress of BiteOpt during optimization
# On an AMD 5950 16 core processor more than one million fitness executions per second
# are performed. 

from scipy.optimize import Bounds
from fcmaes.optimizer import wrapper
from evaluate import Env
from bite import Bite
import multiprocessing as mp

def get_bite(vehicle_num, target_num, map_size, evals):
    env = Env(vehicle_num,target_num,map_size,visualized=True)
    return Bite(vehicle_num,env.vehicles_speed,target_num,env.targets,env.time_lim, evals)

def optimize(vehicle_num, target_num, map_size):
    evals = 2000000
    bite = get_bite(vehicle_num, target_num, map_size, evals)
    bite.fitness = wrapper(bite.fitness)
    bite.workers = mp.cpu_count()
    task_assignment, time =  bite.run()
    print(str(task_assignment), str(time))
   
if __name__=='__main__':
    # small scale
    #optimize(5,30,5e3)
    # medium scale
    #optimize(10,60,1e4)
    # large scale
    optimize(15,90,1.5e4)
