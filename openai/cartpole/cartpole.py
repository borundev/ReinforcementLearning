from ..funcs_and_wrappers import Environment
import numpy as np
import tensorflow as tf
from gym import logger

MAX_ANGLE_DEFAULT=15*np.pi/180.
MAX_DIST_DEFAULT=2.4
MAX_STEPS_DEFAULT=200

class MyCartPole(Environment):
    """
    A wrapper around the Cartpole environment. It takes customized max_angle, max_distance and
    max_steps
    """
    def __init__(self,
                 min_angle= - MAX_ANGLE_DEFAULT,
                 max_angle = MAX_ANGLE_DEFAULT,
                 min_distance = -MAX_DIST_DEFAULT,
                 max_distance = MAX_DIST_DEFAULT,
                 max_steps = MAX_STEPS_DEFAULT):
        super().__init__("CartPole-v1")
        self.min_angle=min_angle
        self.max_angle=max_angle
        self.min_distance=min_distance
        self.max_distance=max_distance
        self.max_steps=max_steps
        self.steps_beyond_done=None

    def reset(self, **kwargs):
        self.steps_beyond_done = None
        return super().reset(**kwargs)

    def step(self, action):
        """
        Performs a step action on the underlying environment but changes the done status
        depending on the max_angle, max_distance and max_steps

        :param action:
        :return:
        """
        obs, reward, done, something =super().step(action)

        cond_angle=self.min_angle < obs[2] < self.max_angle
        cond_dist=self.min_distance < obs[0] < self.max_distance
        cond_steps=self.steps <= self.max_steps

        done_new = not (cond_angle and cond_dist and cond_steps)

        if done_new:
            # print(cond_angle, cond_dist, cond_steps)
            pass

        self.done=done_new

        if not self.done:
            reward = 1.0
        elif self.steps_beyond_done is None:
            # Pole just fell!
            self.steps_beyond_done = 0
            reward = 1.0
        else:
            if self.steps_beyond_done >= 0:
                logger.warn("Calling step even after {} reached state done {} number of "
                            "times".format(self,self.steps_beyond_done))
                print("Calling step even after {} reached state done {}/{} number of "
                            "times".format(self,self.steps_beyond_done,self.steps))
            self.steps_beyond_done += 1
            reward = 0.0

        return obs, reward, done_new, something

    def render(self,mode='human',world_width_factor=5):

        def render(self, mode='human'):
            screen_width = 600
            screen_height = 400

            world_width = self.x_threshold * 2
            scale = screen_width/world_width
            carty = 100  # TOP OF CART
            polewidth = 10.0
            polelen = scale * (2 * self.length)
            cartwidth = 50.0
            cartheight = 30.0

            if self.viewer is None:
                from gym.envs.classic_control import rendering
                self.viewer = rendering.Viewer(screen_width, screen_height)
                l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
                axleoffset = cartheight / 4.0
                cart = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
                self.carttrans = rendering.Transform()
                cart.add_attr(self.carttrans)
                self.viewer.add_geom(cart)
                l, r, t, b = -polewidth / 2, polewidth / 2, polelen - polewidth / 2, -polewidth / 2
                pole = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
                pole.set_color(.8, .6, .4)
                self.poletrans = rendering.Transform(translation=(0, axleoffset))
                pole.add_attr(self.poletrans)
                pole.add_attr(self.carttrans)
                self.viewer.add_geom(pole)
                self.axle = rendering.make_circle(polewidth / 2)
                self.axle.add_attr(self.poletrans)
                self.axle.add_attr(self.carttrans)
                self.axle.set_color(.5, .5, .8)
                self.viewer.add_geom(self.axle)
                self.track = rendering.Line((0, carty), (screen_width, carty))
                self.track.set_color(0, 0, 0)
                self.viewer.add_geom(self.track)

                self._pole_geom = pole

            if self.state is None:
                return None

            # Edit the pole polygon vertex
            pole = self._pole_geom
            l, r, t, b = -polewidth / 2, polewidth / 2, polelen - polewidth / 2, -polewidth / 2
            pole.v = [(l, b), (l, t), (r, t), (r, b)]

            x = self.state
            cartx = x[0]/world_width_factor * scale + screen_width / 2.0  # MIDDLE OF CART
            self.carttrans.set_translation(cartx, carty)
            self.poletrans.set_rotation(-x[2])

            return self.viewer.render(return_rgb_array=mode == 'rgb_array')
        return render(self.env,mode)

class Agent(object):

    """
    This captures the notion of an agent that interacts with an environment. For now this is
    written specialized to the cartpole environment. When I play with more environments I may
    make this more general.

    """
    def __init__(self,
                 min_angle=- MAX_ANGLE_DEFAULT,
                 max_angle=MAX_ANGLE_DEFAULT,
                 min_distance=-MAX_DIST_DEFAULT,
                 max_distance=MAX_DIST_DEFAULT,
                 max_steps=MAX_STEPS_DEFAULT,
                 seed=None
                 ):

        self.env = MyCartPole(min_angle,max_angle,
                              min_distance,max_distance,
                              max_steps)
        self.obs=None
        self.seed=seed
        self.reset()

    @property
    def max_steps(self):
        return self.env.max_steps

    @property
    def done(self):
        return self.env.done

    @done.setter
    def done(self,done):
        self.env.done=done

    def reset(self):
        if self.seed:
            self.env.seed(self.seed)
        obs = self.env.reset()
        self.obs = obs

    def act(self, action):
        obs, reward, done, info = self.env.step(action)
        self.obs = obs
        return reward

    def play_one_step(self, model, loss_fn):
        """
        This is the actual function where an action is taken based on the idea of policy gradient.

        1. The probablities of actions are taken based on the state of the agent+environment and
        the model.
        2. An action itself is sampled based on the probabilities.
        3. The loss is computed __assuming__ the action is the correct one
        4. Gradients are computed for the above loss

        The function returns the reward and the gradients. In the bigger picture of policy
        graidient, the reward would be normalized and weighted based on the bigger goal and may
        turn out to be negative. It will then be multiplied by the graidients and applied to the
        model

        :param model: The model that acts on the state of the agent+environment
        :param loss_fn: The loss function to be used
        :return: The reward and gradients
        """
        with tf.GradientTape() as tape:
            # get probabilities from the model
            probs = model(self.obs[np.newaxis])
            probs_numpy = probs.numpy()[0]

            # see which category is picked randomly
            action = np.random.choice(len(probs_numpy), p=probs_numpy)
            target = tf.cast(action, tf.int32)

            # compute loss
            loss = tf.reduce_mean(loss_fn(target, probs))
        grads = tape.gradient(loss, model.trainable_variables)
        reward = self.act(action)
        return reward, grads

    def play_multiple_steps(self, model, loss_fn):
        """
        Plays up to self.max_steps if the 'done' flag if the enviroment is not triggered.
        Returns the rewards and gradients.

        :param model:
        :param loss_fn:
        :return:
        """
        current_rewards = []
        current_grads = []
        self.reset()
        for step in range(self.max_steps):
            reward, grads = self.play_one_step(model, loss_fn)
            current_rewards.append(reward)
            current_grads.append(grads)
            if self.done:
                break
        return current_rewards, current_grads

    def play_one_step_predict(self, model):

        """
        Given the state of the agent+environemnt this function uses the model to predict
        probablities of the next move and takes the one with the highest probability. So in
        predict we are using argmax unlike in the learning where we chose an action based on
        probabilties.

        :param model:
        :return:
        """

        # get probabilities from the model
        probs = model.predict(self.obs[np.newaxis])

        # get the choice from argmax
        choice = tf.argmax(probs[0]).numpy()

        reward = self.act(choice)
        return reward, choice

    def play_a_round(self, model):
        """
        Plays a full round (upto self.max_steps if the environment doesn't become 'done' earlier) in prediction mode.
        It returns observations, actions and rendered frames.

        :param model:
        :return:
        """
        obs = []
        choices = []
        self.reset()
        frames = []
        for step in range(self.max_steps):
            obs.append(self.obs)
            frames.append(self.env.render(mode='rgb_array'))
            reward, choice = self.play_one_step_predict(model)
            choices.append(choice)
            if self.done:
                break
        return np.stack(obs), np.array(choices).reshape(-1,1), frames


class AgentPool(object):

    """
    A pool of agents which acts on their own respective environments. The play_multiple_steps
    method calls the respective play_multiple_steps of all the agents that makes them agents run
    a full round (upto max_steps or till 'done') and the rewards are returned along with the
    normalized and discounted returns and the gradients making updating the weights to the model
    easy.

    """
    def __init__(self, n_agents, discount_rate=.95,
                 min_angle= - MAX_ANGLE_DEFAULT,
                 max_angle = MAX_ANGLE_DEFAULT,
                 min_distance = -MAX_DIST_DEFAULT,
                 max_distance = MAX_DIST_DEFAULT,
                 max_steps = MAX_STEPS_DEFAULT):

        self.n_agents=n_agents
        self.agents = [Agent(min_angle, max_angle,
                             min_distance, max_distance,
                             max_steps) for _  in range(n_agents)]
        self.discount_rate = discount_rate

    @staticmethod
    def discount_rewards(rewards, discount_rate):
        discounted = np.array(rewards)
        for step in range(len(rewards) - 2, -1, -1):
            discounted[step] += discounted[step + 1] * discount_rate
        return discounted

    @staticmethod
    def discount_and_normalize_rewards(all_rewards, discount_rate):
        all_discounted_rewards = [AgentPool.discount_rewards(rewards, discount_rate)
                                  for rewards in all_rewards]
        flat_rewards = np.concatenate(all_discounted_rewards)
        reward_mean = flat_rewards.mean()
        reward_std = flat_rewards.std()
        return [(discounted_rewards - reward_mean) / reward_std
                for discounted_rewards in all_discounted_rewards]

    #def get_probs_single_step(self,model):
    #    model(np.stack([_a.obs for _a in self.agents]))

    def play_multiple_steps(self, model, loss_fn):
        """
        Run a full run on all the agents in the pool, return the rewards, the discounted and
        normalized returns and the gradients.
        :param model:
        :param loss_fn:
        :return:
        """
        all_rewards = []
        all_grads = []
        for agent in self.agents:
            rewards, grads = agent.play_multiple_steps(model, loss_fn)
            all_rewards.append(rewards)
            all_grads.append(grads)
        return all_rewards, AgentPool.discount_and_normalize_rewards(all_rewards,
                                                                     self.discount_rate), all_grads



def train_model_on_agent(agent_pool,model,optimizer,loss_fn,n_iterations,stop_at_reward=None):

    stop_at_next=False

    for iteration in range(n_iterations):

        if stop_at_next:
            break

        all_rewards,all_final_rewards,all_grads = agent_pool.play_multiple_steps(model,loss_fn)
        total_rewards = sum(map(sum, all_rewards))
        print("\rIteration: {}, mean rewards: {:.1f}".format(
            iteration, total_rewards / agent_pool.n_agents), end="")
        if stop_at_reward and total_rewards / agent_pool.n_agents>=stop_at_reward:
            stop_at_next=True

        all_mean_grads = []
        for var_index in range(len(model.trainable_variables)):
            mean_grads = tf.reduce_mean(
                [final_reward * all_grads[episode_index][step][var_index]
                 for episode_index, final_rewards in enumerate(all_final_rewards)
                     for step, final_reward in enumerate(final_rewards)], axis=0)
            all_mean_grads.append(mean_grads)
        optimizer.apply_gradients(zip(all_mean_grads, model.trainable_variables))