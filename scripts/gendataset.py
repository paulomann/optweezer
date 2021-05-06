import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

k_B = 1.38e-23 # Boltzmann cte. in J / K
T = 300 # temperature in Kelvin
dt = 1e-4
max_time = 1
x0 = 0.0
gamma =  1.29926e-9

n_t = 1000 #training sample size
b_size = 100 #batch size do trainamento
epocas = 5 #epocas de treinamento

trace_size = int((max_time/dt)+1)  

spread_min = 20
spread_max = 40

def linear_simulation(t_max, dt, initial_position, k_spring, gamma):
    """
    Simulate the brownian movement of a overdamped system under linear force 
    through Euler-Maruyama method 

    Parameters
    ----------
    t_max            : final simulation time, such that t \in (0, t_max)
    dt               : time increment
    initial_position : initial condition
    k_spring         : spring constant
    g                : damping constant

    Returns
    -------
    save_times : array with time stamps
    positions : array with simulated positions

    """
    

    N_time = int(t_max/dt)    
    # Number of time stamps
    t = np.linspace(0, t_max, N_time)                                                   # Array with time stamps
    
    positions = np.zeros(N_time)                                               # Array to store calculated positions
    positions[0] = initial_position                                             # Define initial condition
    
    w = np.sqrt(2.0 * k_B * T * dt / gamma) * np.random.normal(size=N_time)       # Wiener increment
    
    for i in range(N_time-1):                                                    # Loop through each time stamp and calculate current position
        positions[i+1] = positions[i] - (k_spring/gamma) * positions[i] * dt + w[i] # Euler-Maruyama method 
        
    return positions


def non_linear_simulation(t_max, dt, initial_position, k_spring, k_a, gamma):
    """
    Simulate the brownian movement of a overdamped system under the influence of non-linearity
    through Euler-Maruyama method 

    Parameters
    ----------
    t_max            : final simulation time, such that t \in (0, t_max)
    dt               : time increment
    initial_position : initial condition
    k_spring         : spring constant
    g                : damping constant
    k_a              : non-linear constant

    Returns
    -------
    save_times : array with time stamps
    positions : array with simulated positions

    """
    
    N_time = int(t_max/dt)                                                      # Number of time stamps
    t = np.linspace(0, t_max, N_time)                                                   # Array with time stamps
    
    positions = np.zeros(N_time)                                               # Array to store calculated positions
    positions[0] = initial_position                                             # Define initial condition
    
    w = np.sqrt(2.0 * k_B * T * dt / gamma) * np.random.normal(size=N_time)       # Wiener increment
    
    for i in range(N_time-1):                                                  # Loop through each time stamp and calculate current position
        positions[i+1] = positions[i] - (k_spring/gamma)*positions[i]*dt - (k_a/gamma)*((positions[i])**3)*dt + w[i] # Euler-Maruyama method 
        
    return positions

def create_samples(num_samples):
    
    k_array = np.linspace(spread_min, spread_max, num=int(num_samples/2))
    k_array = k_array * gamma
    linear = np.array([np.append(linear_simulation(max_time,dt,x0,k,gamma), 0.0) for k in k_array])

    non_linear = np.array([np.append(non_linear_simulation(max_time,dt,x0,k,k*0.5*1e14,gamma), 0.0) for k in k_array])
    
    max_value =  max(np.max(linear), np.max(non_linear))
    linear = linear/max_value
    non_linear = non_linear/max_value
    #TODO change

    linear[:,-1]=1.0
    
    linear = np.float32(linear)
    non_linear = np.float32(non_linear)
    
    
    samples = np.concatenate((linear,non_linear), axis=0)
    np.random.shuffle(samples)

    return samples

if __name__ == '__main__':
    samples = create_samples(100)

    train, test = train_test_split(samples,  test_size=0.25, stratify=samples[:,-1])
    val, test = train_test_split(test,  test_size=0.5, stratify=test[:,-1])

    np.save("train.npy", train, allow_pickle=False)
    np.save("val.npy", val, allow_pickle=False)
    np.save("test.npy", test, allow_pickle=False)