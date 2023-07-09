import gym

gym.envs.register(
    id='gym_auction_env-v0',
    entry_point='envs.AuctionEnv:AuctionEnv',
)
