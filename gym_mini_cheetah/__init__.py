from gym.envs.registration import register

register(
    id='mini_cheetah-v0',
    entry_point='gym_mini_cheetah.envs:MiniCheetahEnv',
)

register(
    id='mini_cheetah-v1',
    entry_point='gym_mini_cheetah.envs:MiniCheetahEnv1',
)