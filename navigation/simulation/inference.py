import numpy as np
import utils
import time
import config as c
from environment.window import Window
from environment.log import Log
from simulation.brain import Brain


# Define inference class
class Inference(Window):
    def __init__(self):
        super().__init__()

        # Initialize brain
        self.brain = Brain()
        self.init_sim()
        self.brain.init_belief(self.body.pos, self.objects.obj_pos)

        # Initialize error tracking
        self.log = Log()
        self.time = time.time()

    def update(self, dt):
        # Track log
        self.log.track(self.trial, self.step, self.brain,
                       self.body, self.objects)

        # Get observations
        O = [self.get_prop_obs(), self.get_visual_obs()]

        # Perform free energy step
        action = self.brain.inference_step(O, self.step)

        # Update body
        action_noise = utils.add_gaussian_noise(action, c.w_a)
        self.body.update(action_noise)

        # Update objects
        self.objects.move_targets()

        # Print info
        if (self.step + 1) % 100 == 0:
            utils.print_info(self.trial, c.n_trials, self.step, c.n_steps)

        # Stop simulation
        self.step += 1
        if self.step == c.n_steps:
            self.reset_trial()

    def reset_trial(self):
        # Simulation done
        if (self.trial + 1) == c.n_trials:
            utils.print_score(self.log)
            print('\nTime elapsed: {:.2f}s'.format(time.time() - self.time))
            self.log.save_log()
            self.stop()

        # Reset simulation
        else:
            self.step = 0
            self.trial += 1

            self.init_sim()
            self.brain.init_belief(self.body.pos, self.objects.obj_pos)
