# do 'pip install fcmaes --upgrade' before executing this code

# Applies a standard continuous optimization algorithm using the same fitness function as GA.py. 
# The fitness function uses numba for speed up. 

# Executing this file you may monitor the progress of fcmaes continous optimization 
# algorithms during optimization
# On an AMD 5950 16 core processor more than one million fitness executions per second
# can be performed. 

from fcmaes.optimizer import wrapper
from evaluate import Env
import multiprocessing as mp
from fcmaesopt import Optimizer
from fcmaes.optimizer import Bite_cpp, De_cpp, Crfmnes_cpp

def get_optimizer(vehicle_num, target_num, map_size, opt):
    env = Env(vehicle_num,target_num,map_size,visualized=True)
    return Optimizer(vehicle_num,env.vehicles_speed,target_num,env.targets,env.time_lim, opt)

def optimize(vehicle_num, target_num, map_size):
    evals = 2000000
    dim = vehicle_num + target_num - 2
    optimizer = get_optimizer(vehicle_num, target_num, map_size, Bite_cpp(evals))
    #optimizer = get_optimizer(vehicle_num, target_num, map_size, Crfmnes_cpp(evals))
    #optimizer = get_optimizer(vehicle_num, target_num, map_size, De_cpp(evals, ints=[True]*dim))
    optimizer.fitness = wrapper(optimizer.fitness)
    optimizer.workers = mp.cpu_count()
    task_assignment, time =  optimizer.run()
    print(str(task_assignment), str(time))
   
if __name__=='__main__':
    # small scale
    #optimize(5,30,5e3)
    # medium scale
    #optimize(10,60,1e4)
    # large scale
    optimize(15,90,1.5e4)
