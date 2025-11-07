import numpy as np
import config as c


# Define body class
class Body:
    def __init__(self):
        # Initialize body parameters
        self.pos = np.array(c.eta_x_pos)
        self.vel = np.zeros_like(self.pos)

    # Update body with velocity
    def update(self, vel):
        self.vel = np.array(vel)
        self.pos = self.pos + c.dt * self.vel

    # Reset body position
    def set_position(self, pos):
        self.pos = np.array(pos)
        self.vel = np.zeros_like(self.pos)
