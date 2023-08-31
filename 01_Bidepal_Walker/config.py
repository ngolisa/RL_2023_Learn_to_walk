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
        self.gamma = 0.9
        self.buffer_size = 512
        self.batch_size = 128

        self.sync_every = 20000

        self.max_steps = 100
        self.episodes = 1000

        # Agent learning parameters
        self.exploration_rate = .8
        self.exploration_rate_min = 0.2
        self.exploration_rate_decay = .9999999


CFG = Configuration()
