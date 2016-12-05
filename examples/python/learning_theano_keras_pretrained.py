#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import print_function
from vizdoom import *
import itertools as it
from random import sample, randint, random
from time import time, sleep
import numpy as np
import skimage.color, skimage.transform
from tqdm import trange

from keras.layers import Input, Dense, Convolution2D, Activation, Flatten
from keras.models import Model
from keras.optimizers import SGD
import cv2

# Q-learning settings
learning_rate = 0.00025
# learning_rate = 0.0001
discount_factor = 0.99
epochs = 20
learning_steps_per_epoch = 2000
replay_memory_size = 10000
# NN learning settings
batch_size = 64
# Training regime
test_episodes_per_epoch = 100
# Other parameters
frame_repeat = 12
resolution = (224, 224)
episodes_to_watch = 10

model_savefile = "./weights.dump"
# Configuration file path
config_file_path = "../../scenarios/simpler_basic.cfg"


# config_file_path = "../../scenarios/rocket_basic.cfg"
# config_file_path = "../../scenarios/basic.cfg"

# Converts and downsamples the input image
def preprocess(img):
    img = skimage.transform.resize(img, resolution)
    img = img.astype(np.float32)
    return img


class ReplayMemory:
    def __init__(self, capacity):
        state_shape = (capacity, 4096)
        self.s1 = np.zeros(state_shape, dtype=np.float32)
        self.s2 = np.zeros(state_shape, dtype=np.float32)
        self.a = np.zeros(capacity, dtype=np.int32)
        self.r = np.zeros(capacity, dtype=np.float32)
        self.isterminal = np.zeros(capacity, dtype=np.bool_)

        self.capacity = capacity
        self.size = 0
        self.pos = 0

    def add_transition(self, s1, action, s2, isterminal, reward):
        self.s1[self.pos, :] = s1
        self.a[self.pos] = action
        if not isterminal:
            self.s2[self.pos, :] = s2
        self.isterminal[self.pos] = isterminal
        self.r[self.pos] = reward

        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def get_sample(self, sample_size):
        i = sample(range(0, self.size), sample_size)
        return self.s1[i], self.a[i], self.s2[i], self.isterminal[i], self.r[i]


def create_network(available_actions_count):
    input_state = Input(shape=(4096,))
    dense = Dense(output_dim=128, init='glorot_normal')(input_state)
    dense = Activation('relu')(dense)
    output = Dense(output_dim=available_actions_count, init='glorot_normal')(dense)

    model = Model(input=input_state, output=output)
    sgd = SGD(lr=0.00025, momentum=0.9, decay=0.005, nesterov=True)
    model.compile(optimizer=sgd, loss='mse')
    return model


def get_pre_trained_features(im):
    im = 255 * np.stack([im, im, im], axis=0)
    im[0,:,:] -= 103.939  #TODO: Check if the format is RGB or BGR
    im[1,:,:] -= 116.779
    im[2,:,:] -= 123.68
    im = np.expand_dims(im, axis=0)
    features = get_penaltimate_features([im, 0])[0]
    s1new = np.zeros(shape=(1, 4096))
    s1new[0, :] = features
    return s1new

from keras import backend as K
from vggModel import *
CNN_model = vggModel('/home/monaj/Documents/myKeras/LRCN/vgg16_weights.h5')
get_penaltimate_features = K.function([CNN_model.layers[0].input, K.learning_phase()], [CNN_model.layers[-2].output])


def perform_learning_step(epoch, loss):
    """ Makes an action according to eps-greedy policy, observes the result
    (next state, reward) and learns from the transition"""

    def exploration_rate(epoch):
        """# Define exploration rate change over time"""
        start_eps = 1.0
        end_eps = 0.1
        const_eps_epochs = 0.1 * epochs  # 10% of learning time
        eps_decay_epochs = 0.6 * epochs  # 60% of learning time

        if epoch < const_eps_epochs:
            return start_eps
        elif epoch < eps_decay_epochs:
            # Linear decay
            return start_eps - (epoch - const_eps_epochs) / \
                               (eps_decay_epochs - const_eps_epochs) * (start_eps - end_eps)
        else:
            return end_eps

    s1 = preprocess(game.get_state().screen_buffer)
    s1 = get_pre_trained_features(s1)

    # With probability eps make a random action.
    eps = exploration_rate(epoch)
    if random() <= eps:
        a = randint(0, len(actions) - 1)
    else:
        # Choose the best action according to the network.
        q = model.predict(s1)
        a = np.argmax(q)
    reward = game.make_action(actions[a], frame_repeat)

    isterminal = game.is_episode_finished()
    s2 = get_pre_trained_features(preprocess(game.get_state().screen_buffer)) if not isterminal else None

    # Remember the transition that was just experienced.
    memory.add_transition(s1, a, s2, isterminal, reward)
    # Get a random minibatch from the replay memory and learns from it.
    if memory.size > batch_size:
        s1, a, s2, isterminal, r = memory.get_sample(batch_size)
        q1 = model.predict(s1)
        q2 = model.predict(s2)
        isterminal = isterminal.astype(float)
        targets = q1
        targets[range(batch_size), a] = r + (1. - isterminal) * discount_factor * np.max(q2, axis=1)
        loss += model.train_on_batch(s1, targets)
    return loss


# Creates and initializes ViZDoom environment.
def initialize_vizdoom(config_file_path):
    print("Initializing doom...")
    game = DoomGame()
    game.load_config(config_file_path)
    game.set_window_visible(False)
    game.set_mode(Mode.PLAYER)
    game.set_screen_format(ScreenFormat.GRAY8)
    game.set_screen_resolution(ScreenResolution.RES_640X480)
    game.init()
    print("Doom initialized.")
    return game


# Create Doom instance
game = initialize_vizdoom(config_file_path)
# Action = which buttons are pressed
n = game.get_available_buttons_size()
actions = [list(a) for a in it.product([0, 1], repeat=n)]
# Create replay memory which will store the transitions
memory = ReplayMemory(capacity=replay_memory_size)

model = create_network(len(actions))

print("Starting the training!")
time_start = time()
for epoch in range(epochs):
    print("\nEpoch %d\n-------" % (epoch + 1))
    train_episodes_finished = 0
    train_scores = []

    print("Training...")
    loss = 0
    game.new_episode()
    for learning_step in trange(learning_steps_per_epoch):
        loss = perform_learning_step(epoch, loss)
        if game.is_episode_finished():
            score = game.get_total_reward()
            train_scores.append(score)
            game.new_episode()
            train_episodes_finished += 1

    print("%d training episodes played." % train_episodes_finished)
    train_scores = np.array(train_scores)
    print("Results: mean: %.1f±%.1f," % (train_scores.mean(), train_scores.std()),
          "min: %.1f," % train_scores.min(), "max: %.1f," % train_scores.max(), "loss: %f," % loss)

    print("\nTesting...")
    test_episode = []
    test_scores = []
    for test_episode in trange(test_episodes_per_epoch):
        game.new_episode()
        while not game.is_episode_finished():
            state = preprocess(game.get_state().screen_buffer)
            state = get_pre_trained_features(state)
            best_action_index = np.argmax(model.predict(state))

            game.make_action(actions[best_action_index], frame_repeat)
        r = game.get_total_reward()
        test_scores.append(r)

    test_scores = np.array(test_scores)
    print("Results: mean: %.1f±%.1f," % (
        test_scores.mean(), test_scores.std()), "min: %.1f" % test_scores.min(), "max: %.1f" % test_scores.max())

    print("Saving the network weigths to:", model_savefile)
    model.save_weights(model_savefile, overwrite=True)
    print("Total elapsed time: %.2f minutes" % ((time() - time_start) / 60.0))

game.close()
print("======================================")
print("Training finished. It's time to watch!")

print("Loading the network weigths from:", model_savefile)
# Load the network's parameters from a file
model.load_weights(model_savefile)

# Reinitialize the game with window visible
game.set_window_visible(True)
game.set_mode(Mode.ASYNC_PLAYER)
game.init()

for _ in range(episodes_to_watch):
    game.new_episode()
    while not game.is_episode_finished():
        state = preprocess(game.get_state().screen_buffer)
        state = get_pre_trained_features(state)
        best_action_index = np.argmax(model.predict(state))

        # Instead of make_action(a, frame_repeat) in order to make the animation smooth
        game.set_action(actions[best_action_index])
        for _ in range(frame_repeat):
            game.advance_action()

    # Sleep between episodes
    sleep(1.0)
    score = game.get_total_reward()
    print("Total score: ", score)