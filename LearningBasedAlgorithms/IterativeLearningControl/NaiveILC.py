import numpy as np
import matplotlib.pyplot as plt

def ilc_step(target_trajectory, previous_trajectory, previous_input, learning_rate):
    num_samples = len(target_trajectory)
    current_input = np.zeros(num_samples)

    for t in range(1, num_samples):
        error = target_trajectory - previous_trajectory
        current_input[t-1] = previous_input[t-1] + learning_rate * error[t]

    return current_input

def iterative_learning_control(target_trajectory, learning_rate, num_steps, system_dynamics):
    '''
    Naive iterative learning control
    '''
    num_samples = len(target_trajectory)
    current_trajectory = np.zeros(num_samples)
    current_input = np.zeros(num_samples)
    input_records = [current_input,]
    trajectory_records = [current_trajectory,]

    for step in range(num_steps):
        current_input = ilc_step(target_trajectory, current_trajectory, 
                                      current_input, learning_rate)
        current_trajectory = system_dynamics(current_input)
        input_records.append(current_input)
        trajectory_records.append(current_trajectory)
        
    return input_records, trajectory_records

if __name__ == "__main__":
    def system_dynamics(x):
        return np.sin(x)
    
    time_steps = np.linspace(0, 2*np.pi, 200)
    target_trajectory = np.sin(time_steps)

    learning_rate = 0.5
    num_steps = 10
    inputs, trajectories = iterative_learning_control(target_trajectory, learning_rate, num_steps, system_dynamics)

    plt.plot(time_steps, target_trajectory, label="target")
    plt.plot(time_steps, trajectories[-1], label="real")
    plt.legend()

    plt.show()
