import config as c
import numpy as np


# Define objects class
class Objects:
    def __init__(self):
        self.obj_pos = np.zeros((c.n_objects, 2))
        self.obj_dirs = np.zeros((c.n_objects, 2))

        self.sample_targets()

    # Sample targets randomly
    def sample_targets(self):
        for o in range(4):
            # Sample position
            if c.t_pos[o] is None:
                x = np.random.uniform(-c.width / 2, c.width / 2)
                y = np.random.uniform(-c.height / 2, c.height / 2)
                self.obj_pos[o] = x, y
            else:
                self.obj_pos[o] = c.t_pos[o]

            # Sample velocity
            if c.t_dir[o] is None:
                angle = np.random.rand() * 2 * np.pi
                self.obj_dirs[o] = np.array((np.cos(angle), np.sin(angle)))
            else:
                self.obj_dirs[o] = c.t_dir[o]

    # Move targets
    def move_targets(self):
        for o in range(4):
            self.obj_pos[o] += c.t_vel[o] * self.obj_dirs[o]

            # Bounce
            pos_w = self.obj_pos[o] + [c.off_x, c.off_y]
            if not (-c.width / 2 + c.t_size < pos_w[0] <
                    c.width / 2 - c.t_size):
                self.obj_dirs[o, 0] = -self.obj_dirs[o, 0]
            if not (-c.height / 2 + c.t_size < pos_w[1] <
                    c.height / 2 - c.t_size):
                self.obj_dirs[o, 1] = -self.obj_dirs[o, 1]
