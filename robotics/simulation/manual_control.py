import numpy as np
from pyglet.window import key
import utils
import config as c
from environment.window import Window


# Define manual control class
class ManualControl(Window):
    def __init__(self):
        super().__init__()

    def update(self, dt):
        # Get action from user
        action = self.get_pressed()

        # Update body
        self.body.update(np.array(action) * 10)

        # Update objects
        self.objects.move_targets()

        # Print info
        if (self.step + 1) % 100 == 0:
            utils.print_info(self.trial, c.n_trials, self.step, c.n_steps)

        # Reset trial
        self.step += 1
        if self.step == c.n_steps:
            self.reset_trial()

    def reset_trial(self):
        # Simulation done
        if (self.trial + 1) == c.n_trials:
            self.stop()

        # Reset simulation
        else:
            self.init_sim()

            self.step = 0
            self.trial += 1

    # Get action from user input
    def get_pressed(self):
        return [(key.RIGHT in self.keys) - (key.LEFT in self.keys),
                (key.UP in self.keys) - (key.DOWN in self.keys),
                (key.Z in self.keys) - (key.X in self.keys),
                (key.A in self.keys) - (key.S in self.keys)]
