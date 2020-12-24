##Gym Environment for MIT Mini Cheetah
###Introduction
This package consists of a customized gym environment for MIT Mini Cheetah created using PyBullet. It currently supports Stable Baselines 3 and Augmented Random Search (ARS) to train Reinforcement Learning (RL) agents for locomotion on flat terrain. A few pretrained RL agents with tuned hyperparameters are demonstrated.

### Getting Started:
To install the package and its dependencies run the following command, inside the folder, MiniCheetahRL:        
                
        pip install -e .

The code base was tested with gym (0.17.1), PyBullet (3.0.4) with a python version of 3.6.9. However, it is expected to support any future versions of these packages, though they haven't been tested.

### Verify Gym Environment :
        
        cd gym_mini_cheetah
        python3 makeEnv.py

It will output "successfully created" message upon correct installation.

### Test Pretrained Agents :
####1. PPO
        
        cd gym_mini_cheetah/agents/PPO/
        python3 test_PPO.py

####2. ARS
        
        cd gym_mini_cheetah/agents/ARS/
        python3 test_ARS.py

### Train Agents :
####1. PPO
        
        cd gym_mini_cheetah/agents/PPO/
        python3 train_PPO.py

####2. ARS
        
        cd gym_mini_cheetah/agents/ARS/
        

