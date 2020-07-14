from gym.envs.registration import register

register(
    id='VibratingPendulum-v0',
    entry_point='gym_custom_envs.envs:VibPenEnv',
)
register(
    id='DoubleIntegrator-v0',
    entry_point='gym_custom_envs.envs:DoubIntEnv',
)
register(
    id='Quadrotor2D-v0',
    entry_point='gym_custom_envs.envs:Quad2DEnv_v0',
)
register(
    id='CartPoleContinuous-v0',
    entry_point='gym_custom_envs.envs:CartPoleContiEnv',
)
register(
    id='Quadrotor2D-trajopt-v0',
    entry_point='gym_custom_envs.envs:Quad2DEnv_trajopt',
)
