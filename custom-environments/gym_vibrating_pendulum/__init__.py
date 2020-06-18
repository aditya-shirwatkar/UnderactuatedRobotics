from gym.envs.registration import register

register(
    id='VibratingPendulum-v0',
    entry_point='gym_vibrating_pendulum.envs:VibPenEnv',
)
