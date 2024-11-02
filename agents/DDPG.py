import random
from collections import deque

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from utils import Portfolio

# TensorFlow GPU configuration (enables memory growth to avoid full GPU allocation at once)
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

HIDDEN1_UNITS = 24
HIDDEN2_UNITS = 48
HIDDEN3_UNITS = 24


class ActorNetwork:
    def __init__(self, state_size, action_dim, buffer_size, tau, learning_rate, is_eval=False, model_name=""):
        self.tau = tau
        self.learning_rate = learning_rate
        self.action_dim = action_dim
        self.buffer_size = buffer_size

        # Model and target model
        self.model, self.states = self.create_actor_network(state_size, action_dim)
        if is_eval:
            self.model.load_weights(f'saved_models/{model_name}_actor.h5')
        else:
            self.model_target, _ = self.create_actor_network(state_size, action_dim)
            self.model_target.set_weights(self.model.get_weights())  # initialize target network

    def train(self, states_batch, action_grads_batch):
        # Ensure action_grads_batch is not None
        if action_grads_batch is None:
            raise ValueError("Received None for action_grads_batch; ensure gradient computation is correct.")
        
        with tf.GradientTape() as tape:
            actions = self.model(states_batch, training=True)
            # Mean policy gradient for actor update
            sampled_policy_grad = -tf.reduce_mean(action_grads_batch * actions) / self.buffer_size
        grads = tape.gradient(sampled_policy_grad, self.model.trainable_variables)

        # Check if gradients are None
        if any(g is None for g in grads):
            raise ValueError("Received None in gradients; check computation for sampled_policy_grad.")
        
        # Apply gradients
        self.model.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

    def train_target(self):
        actor_weights = self.model.get_weights()
        actor_target_weights = self.model_target.get_weights()
        for i in range(len(actor_weights)):
            actor_target_weights[i] = self.tau * actor_weights[i] + (1 - self.tau) * actor_target_weights[i]
        self.model_target.set_weights(actor_target_weights)

    def create_actor_network(self, state_size, action_dim):
        states = Input(shape=[state_size])
        h0 = Dense(HIDDEN1_UNITS, activation='relu')(states)
        h1 = Dense(HIDDEN2_UNITS, activation='relu')(h0)
        h2 = Dense(HIDDEN3_UNITS, activation='relu')(h1)
        actions = Dense(action_dim, activation='softmax')(h2)
        model = Model(inputs=states, outputs=actions)
        model.compile(optimizer=Adam(learning_rate=self.learning_rate))
        return model, states


class CriticNetwork:
    def __init__(self, state_size, action_dim, tau, learning_rate, is_eval=False, model_name=""):
        self.tau = tau
        self.learning_rate = learning_rate
        self.action_dim = action_dim

        # Model and target model
        self.model, self.actions, self.states = self.create_critic_network(state_size, action_dim)
        if is_eval:
            self.model.load_weights(f'saved_models/{model_name}_critic.h5')
        else:
            self.model_target, _, _ = self.create_critic_network(state_size, action_dim)

    def gradients(self, states_batch, actions_batch):
        # Ensure inputs are tensors
        states_batch = tf.convert_to_tensor(states_batch, dtype=tf.float32)
        actions_batch = tf.convert_to_tensor(actions_batch, dtype=tf.float32)

        with tf.GradientTape() as tape:
            tape.watch(actions_batch)
            Q_values = self.model([states_batch, actions_batch], training=True)
        
        # Print debugging information
        print("states_batch shape:", states_batch.shape)
        print("actions_batch shape:", actions_batch.shape)
        print("Q_values shape:", Q_values.shape)

        # Calculate gradients
        grads = tape.gradient(Q_values, actions_batch)

        # Check if gradients are None
        if grads is None:
            raise ValueError("Gradients are None; check Q-values computation and ensure shapes are compatible.")
        return grads
    
    def train_target(self):
        critic_weights = self.model.get_weights()
        critic_target_weights = self.model_target.get_weights()
        for i in range(len(critic_weights)):
            critic_target_weights[i] = self.tau * critic_weights[i] + (1 - self.tau) * critic_target_weights[i]
        self.model_target.set_weights(critic_target_weights)

    def create_critic_network(self, state_size, action_dim):
        states = Input(shape=[state_size])
        actions = Input(shape=[action_dim])
        h0 = Concatenate()([states, actions])
        h1 = Dense(HIDDEN1_UNITS, activation='relu')(h0)
        h2 = Dense(HIDDEN2_UNITS, activation='relu')(h1)
        h3 = Dense(HIDDEN3_UNITS, activation='relu')(h2)
        Q = Dense(1, activation='linear')(h3)  # Output a single Q-value
        model = Model(inputs=[states, actions], outputs=Q)
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate, decay=1e-6))
        return model, actions, states


class OUNoise:
    def __init__(self, action_dim, mu=0.0, theta=0.15, max_sigma=0.3, min_sigma=0.3, decay_period=100000):
        self.mu = mu
        self.theta = theta
        self.sigma = max_sigma
        self.max_sigma = max_sigma
        self.min_sigma = min_sigma
        self.decay_period = decay_period
        self.action_dim = action_dim
        self.reset()

    def reset(self):
        self.states = np.ones(self.action_dim) * self.mu

    def evolve_state(self):
        x = self.states
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.action_dim)
        self.states = x + dx
        return self.states

    def get_actions(self, actions, t=0):
        ou_state = self.evolve_state()
        self.sigma = self.max_sigma - (self.max_sigma - self.min_sigma) * min(1.0, t / self.decay_period)
        return np.clip(actions + ou_state, 0, 1)


class Agent(Portfolio):
    def __init__(self, state_dim, balance, is_eval=False, model_name=""):
        super().__init__(balance=balance)
        self.model_type = 'DDPG'
        self.state_dim = state_dim
        self.action_dim = 3  # hold, buy, sell
        self.memory = deque(maxlen=100)
        self.buffer_size = 90

        self.gamma = 0.95  # discount factor
        self.is_eval = is_eval
        self.noise = OUNoise(self.action_dim)
        tau = 0.001
        learning_rate_actor = 0.001
        learning_rate_critic = 0.001

        # Initialize networks
        self.actor = ActorNetwork(state_dim, self.action_dim, self.buffer_size, tau, learning_rate_actor, is_eval, model_name)
        self.critic = CriticNetwork(state_dim, self.action_dim, tau, learning_rate_critic)

        # TensorBoard logging
        self.tensorboard = tf.keras.callbacks.TensorBoard(log_dir='./logs/DDPG_tensorboard', update_freq=90)
        self.tensorboard.set_model(self.critic.model)

    def reset(self):
        self.reset_portfolio()
        self.noise.reset()

    def remember(self, state, actions, reward, next_state, done):
        self.memory.append((state, actions, reward, next_state, done))

    def act(self, state, t):
        actions = self.actor.model.predict(state)[0]
        if not self.is_eval:
            return self.noise.get_actions(actions, t)
        return actions

    def experience_replay(self):
        mini_batch = random.sample(self.memory, self.buffer_size)

        y_batch = []
        for state, actions, reward, next_state, done in mini_batch:
            if not done:
                Q_target_value = self.critic.model_target.predict([next_state, self.actor.model_target.predict(next_state)])
                y = reward + self.gamma * Q_target_value
            else:
                y = reward * np.ones((1, self.action_dim))
            y_batch.append(y)

        # Convert y_batch to a tensor
        y_batch = tf.convert_to_tensor(np.vstack(y_batch), dtype=tf.float32)
        
        # Ensure states_batch and actions_batch are tensors
        states_batch = tf.convert_to_tensor(np.vstack([tup[0] for tup in mini_batch]), dtype=tf.float32)
        actions_batch = tf.convert_to_tensor(np.vstack([tup[1] for tup in mini_batch]), dtype=tf.float32)

        # Update critic by minimizing the loss
        loss = self.critic.model.train_on_batch([states_batch, actions_batch], y_batch)

        # Update actor using policy gradients
        action_grads_batch = self.critic.gradients(states_batch, self.actor.model(states_batch))
        self.actor.train(states_batch, action_grads_batch)

        # Update target networks
        self.actor.train_target()
        self.critic.train_target()
        return loss
