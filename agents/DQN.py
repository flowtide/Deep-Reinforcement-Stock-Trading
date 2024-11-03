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

# reference:
# https://arxiv.org/pdf/1312.5602.pdf
class Agent(Portfolio):
    def __init__(self, state_dim, balance, is_eval=False, model_name=""):
        super().__init__(balance=balance)
        self.model_type = 'DQN'
        self.state_dim = state_dim
        self.action_dim = 3  # hold, buy, sell
        self.memory = deque(maxlen=100)
        self.buffer_size = 60

        self.gamma = 0.95
        self.epsilon = 1.0  # initial exploration rate
        self.epsilon_min = 0.01  # minimum exploration rate
        self.epsilon_decay = 0.995 # decrease exploration rate as the agent becomes good at trading
        self.is_eval = is_eval
        if is_eval:
            model_file = f'saved_models/{model_name}.h5'
            print(f'loading model: {model_file}')
            self.model = load_model(model_file)
        else:
            self.model = self.model()

        self.tensorboard = TensorBoard(log_dir='./logs/DQN_tensorboard', update_freq=90)
        self.tensorboard.set_model(self.model)

    def model(self):
        model = Sequential()
        model.add(Dense(units=64, input_dim=self.state_dim, activation='relu'))
        model.add(Dense(units=32, activation='relu'))
        model.add(Dense(units=8, activation='relu'))
        model.add(Dense(self.action_dim, activation='softmax'))
        model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=0.01))
        return model

    def reset(self):
        self.reset_portfolio()
        self.epsilon = 1.0 # reset exploration rate

    def remember(self, state, actions, reward, next_state, done):
        self.memory.append((state, actions, reward, next_state, done))

    def act(self, state):
        if not self.is_eval and np.random.rand() <= self.epsilon:
            return random.randrange(self.action_dim)
        options = self.model.predict(state, verbose=0)
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
