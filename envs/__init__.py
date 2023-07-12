import gym

gym.envs.register(
    id='gym_dag_env-v0',
    entry_point='envs.DAGEnv:DAGEnv',
)
