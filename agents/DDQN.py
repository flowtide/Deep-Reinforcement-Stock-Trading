import random
from collections import deque

import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam

from utils import Portfolio

import tensorflow as tf
# TensorFlow GPU configuration (enables memory growth to avoid full GPU allocation at once)
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

# references:
# https://arxiv.org/pdf/1802.09477.pdf
# https://arxiv.org/pdf/1509.06461.pdf
# https://papers.nips.cc/paper/3964-double-q-learning.pdf
class Agent(Portfolio):
    def __init__(self, state_dim, balance, is_eval=False, model_name=""):
        super().__init__(balance=balance)
        self.model_type = 'DQN'
        self.state_dim = state_dim
        self.action_dim = 3  # hold, buy, sell
        self.memory = deque(maxlen=100)
        self.buffer_size = 60

        self.tau = 0.0001
        self.gamma = 0.95
        self.epsilon = 1.0  # initial exploration rate
        self.epsilon_min = 0.01  # minimum exploration rate
        self.epsilon_decay = 0.995 # decrease exploration rate as the agent becomes good at trading
        self.is_eval = is_eval

        self.model = load_model(f'saved_models/{model_name}.h5') if is_eval else self.model()
        self.model_target = load_model(f'saved_models/{model_name}_target.h5') if is_eval else self.model
        self.model_target.set_weights(self.model.get_weights()) # hard copy model parameters to target model parameters

        self.tensorboard = TensorBoard(log_dir='./logs/DDQN_tensorboard', update_freq=90)
        self.tensorboard.set_model(self.model)

    def update_model_target(self):
        model_weights = self.model.get_weights()
        model_target_weights = self.model_target.get_weights()
        for i in range(len(model_weights)):
            model_target_weights[i] = self.tau * model_weights[i] + (1 - self.tau) * model_target_weights[i]
        self.model_target.set_weights(model_target_weights)

    def model(self):
        model = Sequential()
        model.add(Dense(units=64, input_dim=self.state_dim, activation='relu'))
        model.add(Dense(units=32, activation='relu'))
        model.add(Dense(units=8, activation='relu'))
        model.add(Dense(self.action_dim, activation='softmax'))
        model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=0.001))
        return model

    def reset(self):
        self.reset_portfolio()
        self.epsilon = 1.0

    def remember(self, state, actions, reward, next_state, done):
        self.memory.append((state, actions, reward, next_state, done))

    def act(self, state):
        if not self.is_eval and np.random.rand() <= self.epsilon:
            return random.randrange(self.action_dim)
        options = self.model.predict(state)
        return np.argmax(options[0])

    def experience_replay(self):
        if len(self.memory) < self.buffer_size:
            return  # Not enough samples to replay

        # Randomly sample a mini-batch from memory
        mini_batch = random.sample(self.memory, self.buffer_size)

        # Initialize arrays for states and targets
        states = np.zeros((self.buffer_size, self.state_dim))
        targets = np.zeros((self.buffer_size, self.action_dim))

        for i, (state, actions, reward, next_state, done) in enumerate(mini_batch):
            state = np.array(state)
            next_state = np.array(next_state)

            # Predict the current Q-values
            target = self.model.predict(state.reshape(1, -1), verbose=0)[0]

            if done:
                target[np.argmax(actions)] = reward
            else:
                # Predict the future Q-values
                Q_future = max(self.model.predict(next_state.reshape(1, -1), verbose=0)[0])
                target[np.argmax(actions)] = reward + self.gamma * Q_future

            states[i] = state
            targets[i] = target

        # Train the model on all states and targets in one batch
        history = self.model.fit(states, targets, epochs=1, verbose=0, batch_size=self.buffer_size)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return history.history['loss'][0]
