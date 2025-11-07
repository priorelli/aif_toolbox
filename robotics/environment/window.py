import numpy as np
import pyglet
from pyglet.shapes import Circle, Rectangle, Arc
import utils
import config as c
from environment.body import Body
from environment.objects import Objects


# Define window class
class Window(pyglet.window.Window):
    def __init__(self):
        super().__init__(c.width, c.height, 'Toolbox',
                         vsync=False)
        # Initialize body and objects
        self.body = Body()
        self.objects = Objects()

        # Initialize simulation variables
        self.step, self.trial, self.success = 0, 0, 0

        self.keys = set()
        self.batch = pyglet.graphics.Batch()
        self.fps_display = pyglet.window.FPSDisplay(self)
        self.offset = (c.width / 2 + c.off_x, c.height / 2 + c.off_y)

        # Set background
        pyglet.gl.glClearColor(1, 1, 1, 1)

    def init_sim(self):
        self.body.set_rotation(c.eta_x_int)
        self.objects.sample_targets()

    def on_key_press(self, sym, mod):
        self.keys.add(sym)

    def on_key_release(self, sym, mod):
        self.keys.remove(sym)

    def on_draw(self):
        self.clear()
        objects = self.draw_screen()
        self.batch.draw()
        self.fps_display.draw()

    # Update function to override
    def update(self, dt):
        pass

    # Run simulation with custom update function
    def run(self):
        if c.fps == 0:
            pyglet.clock.schedule(self.update)
        else:
            pyglet.clock.schedule_interval(self.update, 1 / c.fps)
        pyglet.app.run()

    # Stop simulation
    def stop(self):
        pyglet.app.exit()
        self.close()

    # Draw screen
    def draw_screen(self):
        objects = set()

        # Move coordinates on screen
        targets_w = self.objects.obj_pos + self.offset
        pos_w = self.body.poses[:, :2] + self.offset

        # Draw targets
        colors = [(200, 100, 0), (100, 200, 0), (200, 0, 100), (100, 0, 200)]
        for target_w, color in zip(targets_w, colors):
            objects.add(Circle(*target_w, c.t_size, color=color,
                               batch=self.batch))

        # Draw body
        objects.add(Circle(*self.offset, 20, segments=20,
                           color=(0, 100, 200), batch=self.batch))

        for j in range(c.n_joints):
            objects = self.draw_link(objects, j, pos_w[j + 1])

        return objects

    # Draw link
    def draw_link(self, objects, n, pos):
        length, width = self.body.size[n]

        # Draw link
        link = Rectangle(*pos, length, width,
                         color=(0, 100, 200), batch=self.batch)
        link.anchor_position = (length, width / 2)

        link.rotation = -self.body.poses[n + 1, 2]
        objects.add(link)

        # Draw joint
        objects.add(Circle(*pos, width / 2, segments=20,
                           color=(0, 100, 200), batch=self.batch))

        return objects

    # Get proprioceptive observation
    def get_prop_obs(self):
        prop_noise = utils.add_gaussian_noise(self.body.angles, c.w_s)
        prop_norm = utils.normalize(prop_noise, c.norm_polar)

        return prop_norm

    # Get visual observation
    def get_visual_obs(self):
        print(self.body.get_vel()[-1])
        pos = np.vstack([[self.body.poses[-1, :2]],
                         self.objects.obj_pos])
        obj_vel = np.tile(c.t_vel, (2, 1)).T * self.objects.obj_dirs
        vel = np.vstack([[self.body.get_vel()[-1]], obj_vel])
        # vel = np.vstack([[np.zeros(2)], obj_vel])

        vis_norm = utils.normalize(np.array([pos, vel]), c.norm_cart)
        vis_noise = utils.add_gaussian_noise(vis_norm, c.w_s)

        return vis_noise
