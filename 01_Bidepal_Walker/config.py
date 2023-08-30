"""
Configuration Module.

This module defines a singleton-type configuration class that can be used all across our project. This class can contain any parameter that one may want to change from one simulation run to the other.
"""

import random

class Configuration:
    """
    This configuration class is extremely flexible due to a two-step init process. We only instanciate a single instance of it (at the bottom if this file) so that all modules can import this singleton at load time. The second initialization (which happens in main.py) allows the user to input custom parameters of the config class at execution time.
    """

    def __init__(self):
        """
        Declare types but do not instantiate anything
        """
        self.learning_rate = 0.0001
        self.gamma = 0.09
        self.buffer_size = 128
        self.batch_size = 32

        self.sync_every = 200

        self.max_steps = 200
        self.episodes = 10

        # Agent learning parameters
        self.exploration_rate = .8
        self.exploration_rate_min = 0.05
        self.exploration_rate_decay = 0.9999


CFG = Configuration()
