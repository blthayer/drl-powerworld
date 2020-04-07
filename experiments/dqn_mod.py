"""This module contains some modified versions of DQN classes, methods,
etc. from stable_baselines. The point of the modifications is to ensure
that both during training and testing, each unique action is allowed to
be taken at most once per episode.

This is not the best way of doing things, as upstream changes will be
missed. The alternative would be to fork, refactor, modify, subclass,
etc. For the amount of time I have remaining, that's simply too
involved.

Modified areas will be tagged with:
##################
# MODIFICATION:
<modified code>
##################

The original code will be like so:
# ORIGINAL:
# <original code>
"""
import tensorflow as tf
import tensorflow.contrib.layers as tf_layers
import numpy as np
from gym.spaces import MultiDiscrete
from stable_baselines import DQN
from stable_baselines import logger
from stable_baselines.common import SetVerbosity, TensorboardWriter, tf_util
from stable_baselines.common.vec_env import VecEnv
from stable_baselines.common.schedules import LinearSchedule
from stable_baselines.common.policies import nature_cnn
from stable_baselines.deepq.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer
from stable_baselines.a2c.utils import total_episode_reward_logger
from stable_baselines.deepq.policies import FeedForwardPolicy, DQNPolicy


class DQNUniqueActions(DQN):
    """Subclass with a modified "learn" method which only allows actions
    to be taken once per episode.
    """
    def learn(self, total_timesteps, callback=None, log_interval=100, tb_log_name="DQN",
              reset_num_timesteps=True, replay_wrapper=None):

        new_tb_log = self._init_num_timesteps(reset_num_timesteps)

        with SetVerbosity(self.verbose), TensorboardWriter(self.graph, self.tensorboard_log, tb_log_name, new_tb_log) \
                as writer:
            self._setup_learn()

            # Create the replay buffer
            if self.prioritized_replay:
                self.replay_buffer = PrioritizedReplayBuffer(self.buffer_size, alpha=self.prioritized_replay_alpha)
                if self.prioritized_replay_beta_iters is None:
                    prioritized_replay_beta_iters = total_timesteps
                else:
                    prioritized_replay_beta_iters = self.prioritized_replay_beta_iters
                self.beta_schedule = LinearSchedule(prioritized_replay_beta_iters,
                                                    initial_p=self.prioritized_replay_beta0,
                                                    final_p=1.0)
            else:
                self.replay_buffer = ReplayBuffer(self.buffer_size)
                self.beta_schedule = None

            if replay_wrapper is not None:
                assert not self.prioritized_replay, "Prioritized replay buffer is not supported by HER"
                self.replay_buffer = replay_wrapper(self.replay_buffer)

            # Create the schedule for exploration starting from 1.
            self.exploration = LinearSchedule(schedule_timesteps=int(self.exploration_fraction * total_timesteps),
                                              initial_p=self.exploration_initial_eps,
                                              final_p=self.exploration_final_eps)

            episode_rewards = [0.0]
            episode_successes = []
            obs = self.env.reset()
            reset = True

            ############################################################
            # MODIFICATION:
            # Track list of actions taken each episode. This is
            # intentionally not a set so that we can use np.isin.
            action_list = list()
            ############################################################

            for _ in range(total_timesteps):
                if callback is not None:
                    # Only stop training if return value is False, not when it is None. This is for backwards
                    # compatibility with callbacks that have no return statement.
                    if callback(locals(), globals()) is False:
                        break
                # Take action and update exploration to the newest value
                kwargs = {}
                if not self.param_noise:
                    update_eps = self.exploration.value(self.num_timesteps)
                    update_param_noise_threshold = 0.
                else:
                    update_eps = 0.
                    # Compute the threshold such that the KL divergence between perturbed and non-perturbed
                    # policy is comparable to eps-greedy exploration with eps = exploration.value(t).
                    # See Appendix C.1 in Parameter Space Noise for Exploration, Plappert et al., 2017
                    # for detailed explanation.
                    update_param_noise_threshold = \
                        -np.log(1. - self.exploration.value(self.num_timesteps) +
                                self.exploration.value(self.num_timesteps) / float(self.env.action_space.n))
                    kwargs['reset'] = reset
                    kwargs['update_param_noise_threshold'] = update_param_noise_threshold
                    kwargs['update_param_noise_scale'] = True
                with self.sess.as_default():
                    ####################################################
                    # MODIFICATION:
                    # Rename variable from original, since it's now
                    # going to come back as an array due to the
                    # modified build_act function being used to
                    # construct everything.
                    action_arr = self.act(np.array(obs)[None], update_eps=update_eps, **kwargs)[0]
                    ####################################################
                    # ORIGINAL:
                    # action = self.act(np.array(obs)[None], update_eps=update_eps, **kwargs)[0]

                ########################################################
                # MODIFICATION:
                # Get the best action that has not yet been taken this
                # episode.
                action = \
                    action_arr[np.argmin(np.isin(action_arr, action_list))]
                # Add this action to the list.
                action_list.append(action)
                ########################################################

                env_action = action
                reset = False
                new_obs, rew, done, info = self.env.step(env_action)
                # Store transition in the replay buffer.
                self.replay_buffer.add(obs, action, rew, new_obs, float(done))
                obs = new_obs

                if writer is not None:
                    ep_rew = np.array([rew]).reshape((1, -1))
                    ep_done = np.array([done]).reshape((1, -1))
                    total_episode_reward_logger(self.episode_reward, ep_rew, ep_done, writer,
                                                self.num_timesteps)

                episode_rewards[-1] += rew
                if done:
                    ####################################################
                    # MODIFICATION:
                    # Clear the list.
                    action_list.clear()
                    ####################################################
                    maybe_is_success = info.get('is_success')
                    if maybe_is_success is not None:
                        episode_successes.append(float(maybe_is_success))
                    if not isinstance(self.env, VecEnv):
                        obs = self.env.reset()
                    episode_rewards.append(0.0)
                    reset = True

                # Do not train if the warmup phase is not over
                # or if there are not enough samples in the replay buffer
                can_sample = self.replay_buffer.can_sample(self.batch_size)
                if can_sample and self.num_timesteps > self.learning_starts \
                        and self.num_timesteps % self.train_freq == 0:
                    # Minimize the error in Bellman's equation on a batch sampled from replay buffer.
                    # pytype:disable=bad-unpacking
                    if self.prioritized_replay:
                        assert self.beta_schedule is not None, \
                               "BUG: should be LinearSchedule when self.prioritized_replay True"
                        experience = self.replay_buffer.sample(self.batch_size,
                                                               beta=self.beta_schedule.value(self.num_timesteps))
                        (obses_t, actions, rewards, obses_tp1, dones, weights, batch_idxes) = experience
                    else:
                        obses_t, actions, rewards, obses_tp1, dones = self.replay_buffer.sample(self.batch_size)
                        weights, batch_idxes = np.ones_like(rewards), None
                    # pytype:enable=bad-unpacking

                    if writer is not None:
                        # run loss backprop with summary, but once every 100 steps save the metadata
                        # (memory, compute time, ...)
                        if (1 + self.num_timesteps) % 100 == 0:
                            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                            run_metadata = tf.RunMetadata()
                            summary, td_errors = self._train_step(obses_t, actions, rewards, obses_tp1, obses_tp1,
                                                                  dones, weights, sess=self.sess, options=run_options,
                                                                  run_metadata=run_metadata)
                            writer.add_run_metadata(run_metadata, 'step%d' % self.num_timesteps)
                        else:
                            summary, td_errors = self._train_step(obses_t, actions, rewards, obses_tp1, obses_tp1,
                                                                  dones, weights, sess=self.sess)
                        writer.add_summary(summary, self.num_timesteps)
                    else:
                        _, td_errors = self._train_step(obses_t, actions, rewards, obses_tp1, obses_tp1, dones, weights,
                                                        sess=self.sess)

                    if self.prioritized_replay:
                        new_priorities = np.abs(td_errors) + self.prioritized_replay_eps
                        assert isinstance(self.replay_buffer, PrioritizedReplayBuffer)
                        self.replay_buffer.update_priorities(batch_idxes, new_priorities)

                if can_sample and self.num_timesteps > self.learning_starts and \
                        self.num_timesteps % self.target_network_update_freq == 0:
                    # Update target network periodically.
                    self.update_target(sess=self.sess)

                if len(episode_rewards[-101:-1]) == 0:
                    mean_100ep_reward = -np.inf
                else:
                    mean_100ep_reward = round(float(np.mean(episode_rewards[-101:-1])), 1)

                num_episodes = len(episode_rewards)
                if self.verbose >= 1 and done and log_interval is not None and len(episode_rewards) % log_interval == 0:
                    logger.record_tabular("steps", self.num_timesteps)
                    logger.record_tabular("episodes", num_episodes)
                    if len(episode_successes) > 0:
                        logger.logkv("success rate", np.mean(episode_successes[-100:]))
                    logger.record_tabular("mean 100 episode reward", mean_100ep_reward)
                    logger.record_tabular("% time spent exploring",
                                          int(100 * self.exploration.value(self.num_timesteps)))
                    logger.dump_tabular()

                self.num_timesteps += 1

        return self


def build_act_mod(q_func, ob_space, ac_space, stochastic_ph, update_eps_ph, sess):
    """
    Original from build_graph.py from stable_baselines.deepq. This
    modified version returns the full sorted action array instead of
    the single best action.

    Creates the act function:

    :param q_func: (DQNPolicy) the policy
    :param ob_space: (Gym Space) The observation space of the environment
    :param ac_space: (Gym Space) The action space of the environment
    :param stochastic_ph: (TensorFlow Tensor) the stochastic placeholder
    :param update_eps_ph: (TensorFlow Tensor) the update_eps placeholder
    :param sess: (TensorFlow session) The current TensorFlow session
    :return: (function (TensorFlow Tensor, bool, float): TensorFlow Tensor, (TensorFlow Tensor, TensorFlow Tensor)
        act function to select and action given observation (See the top of the file for details),
        A tuple containing the observation placeholder and the processed observation placeholder respectively.
    """
    eps = tf.get_variable("eps", (), initializer=tf.constant_initializer(0))

    policy = q_func(sess, ob_space, ac_space, 1, 1, None)
    obs_phs = (policy.obs_ph, policy.processed_obs)
    ####################################################################
    # MODIFICATION:
    # Get all sorted q_values instead of just the best one. Have to
    # cast to int64, since it comes back as int32 which is incompatible
    # with the random_actions, which is int64.
    deterministic_actions = tf.cast(
        tf.argsort(policy.q_values, axis=1, direction='DESCENDING'), tf.int64)
    ####################################################################
    # ORIGINAL:
    # Note that the original comes out as int64 since that's the default
    # from argmax.
    # deterministic_actions = tf.argmax(policy.q_values, axis=1)

    batch_size = tf.shape(policy.obs_ph)[0]
    n_actions = ac_space.nvec if isinstance(ac_space, MultiDiscrete) else ac_space.n
    ####################################################################
    # MODIFICATION:
    # TODO: It feels wrong not to use "batch_size" like was done in the
    #   original, but this seems to be working.
    random_actions = tf.random_uniform((1, n_actions), minval=0, maxval=n_actions, dtype=tf.int64)
    ####################################################################
    # ORIGINAL:
    # random_actions = tf.random_uniform(tf.stack([batch_size]), minval=0, maxval=n_actions, dtype=tf.int64)
    chose_random = tf.random_uniform(tf.stack([batch_size]), minval=0, maxval=1, dtype=tf.float32) < eps
    stochastic_actions = tf.where(chose_random, random_actions, deterministic_actions)

    output_actions = tf.cond(stochastic_ph, lambda: stochastic_actions, lambda: deterministic_actions)
    update_eps_expr = eps.assign(tf.cond(update_eps_ph >= 0, lambda: update_eps_ph, lambda: eps))
    _act = tf_util.function(inputs=[policy.obs_ph, stochastic_ph, update_eps_ph],
                            outputs=output_actions,
                            givens={update_eps_ph: -1.0, stochastic_ph: True},
                            updates=[update_eps_expr])

    def act(obs, stochastic=True, update_eps=-1):
        return _act(obs, stochastic, update_eps)

    return act, obs_phs


def step_mod(self: FeedForwardPolicy, obs, state=None, mask=None,
             deterministic=True):
    """Originally from dqn.FeedForwardPolicy.step. This modified method
    will return the full sorted action array instead of the single best.
    """
    q_values, actions_proba = self.sess.run([self.q_values, self.policy_proba], {self.obs_ph: obs})
    if deterministic:
        ################################################################
        # MODIFIED:
        actions = np.flip(np.argsort(q_values))
        ################################################################
        # ORIGINAL:
        # actions = np.argmax(q_values, axis=1)
    else:
        # Unefficient sampling
        # TODO: replace the loop
        # maybe with Gumbel-max trick ? (http://amid.fish/humble-gumbel)
        actions = np.zeros((len(obs),), dtype=np.int64)
        for action_idx in range(len(obs)):
            actions[action_idx] = np.random.choice(self.n_actions, p=actions_proba[action_idx])

    return actions, q_values, None

class FeedForwardPolicy(DQNPolicy):
    """
    Policy object that implements a DQN policy, using a feed forward neural network.

    :param sess: (TensorFlow session) The current TensorFlow session
    :param ob_space: (Gym Space) The observation space of the environment
    :param ac_space: (Gym Space) The action space of the environment
    :param n_env: (int) The number of environments to run
    :param n_steps: (int) The number of steps to run for each environment
    :param n_batch: (int) The number of batch to run (n_envs * n_steps)
    :param reuse: (bool) If the policy is reusable or not
    :param layers: ([int]) The size of the Neural network for the policy (if None, default to [64, 64])
    :param cnn_extractor: (function (TensorFlow Tensor, ``**kwargs``): (TensorFlow Tensor)) the CNN feature extraction
    :param feature_extraction: (str) The feature extraction type ("cnn" or "mlp")
    :param obs_phs: (TensorFlow Tensor, TensorFlow Tensor) a tuple containing an override for observation placeholder
        and the processed observation placeholder respectively
    :param layer_norm: (bool) enable layer normalisation
    :param dueling: (bool) if true double the output MLP to compute a baseline for action scores
    :param act_fun: (tf.func) the activation function to use in the neural network.
    :param kwargs: (dict) Extra keyword arguments for the nature CNN feature extraction
    """

    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=False, layers=None,
                 cnn_extractor=nature_cnn, feature_extraction="cnn",
                 obs_phs=None, layer_norm=False, dueling=True, act_fun=tf.nn.relu, **kwargs):
        super(FeedForwardPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps,
                                                n_batch, dueling=dueling, reuse=reuse,
                                                scale=(feature_extraction == "cnn"), obs_phs=obs_phs)

        self._kwargs_check(feature_extraction, kwargs)

        if layers is None:
            layers = [64, 64]

        with tf.variable_scope("model", reuse=reuse):
            with tf.variable_scope("action_value"):
                if feature_extraction == "cnn":
                    extracted_features = cnn_extractor(self.processed_obs, **kwargs)
                    action_out = extracted_features
                else:
                    extracted_features = tf.layers.flatten(self.processed_obs)
                    action_out = extracted_features
                    for layer_size in layers:
                        action_out = tf_layers.fully_connected(action_out, num_outputs=layer_size, activation_fn=None)
                        if layer_norm:
                            action_out = tf_layers.layer_norm(action_out, center=True, scale=True)
                        action_out = act_fun(action_out)

                action_scores = tf_layers.fully_connected(action_out, num_outputs=self.n_actions, activation_fn=None)

            if self.dueling:
                with tf.variable_scope("state_value"):
                    state_out = extracted_features
                    for layer_size in layers:
                        state_out = tf_layers.fully_connected(state_out, num_outputs=layer_size, activation_fn=None)
                        if layer_norm:
                            state_out = tf_layers.layer_norm(state_out, center=True, scale=True)
                        state_out = act_fun(state_out)
                    state_score = tf_layers.fully_connected(state_out, num_outputs=1, activation_fn=None)
                action_scores_mean = tf.reduce_mean(action_scores, axis=1)
                action_scores_centered = action_scores - tf.expand_dims(action_scores_mean, axis=1)
                q_out = state_score + action_scores_centered
            else:
                q_out = action_scores

        self.q_values = q_out
        self._setup_init()

    def step(self, obs, state=None, mask=None, deterministic=True):
        q_values, actions_proba = self.sess.run([self.q_values, self.policy_proba], {self.obs_ph: obs})
        if deterministic:
            actions = np.argmax(q_values, axis=1)
        else:
            # Unefficient sampling
            # TODO: replace the loop
            # maybe with Gumbel-max trick ? (http://amid.fish/humble-gumbel)
            actions = np.zeros((len(obs),), dtype=np.int64)
            for action_idx in range(len(obs)):
                actions[action_idx] = np.random.choice(self.n_actions, p=actions_proba[action_idx])

        return actions, q_values, None

    def proba_step(self, obs, state=None, mask=None):
        return self.sess.run(self.policy_proba, {self.obs_ph: obs})