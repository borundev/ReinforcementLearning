import gym


class Environment(gym.Wrapper):
    """
    A wrapper around OpenAI gym Wrapper which itself is a wrapper around environments. This is
    very shallow and the only purpose is to init with the name.
    """

    def __init__(self, env_name):
        super().__init__(gym.make(env_name))
        self.done = False

    def reset(self, **kwargs):
        self.steps = 0
        self.done = False
        return super().reset(**kwargs)

    def step(self, action):
        self.steps += 1
        return super().step(action)
