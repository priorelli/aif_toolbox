import numpy as np
import utils
import config as c


# Define log class
class Log:
    def __init__(self):
        # Initialize logs
        self.angles = np.zeros((c.n_steps, c.n_joints))
        self.est_angles = np.zeros_like(self.angles)

        self.pos = np.zeros((c.n_steps, c.n_joints + 1, 2))
        self.est_pos = np.zeros_like(self.pos)

        self.targets_pos = np.zeros((c.n_trials, c.n_steps, c.n_objects, 2))
        self.est_targets_pos = np.zeros_like(self.targets_pos)

        self.states = np.zeros((c.n_trials, c.n_steps, c.n_objects))
        self.disc_obs = np.zeros((c.n_trials, c.n_steps, c.n_objects))

        self.L_int = np.zeros((c.n_trials, c.n_steps, c.n_objects))
        self.L_ext = np.zeros_like(self.L_int)

    # Track logs for each iteration
    def track(self, trial, step, brain, body, objects):
        self.angles[step] = body.angles
        prop_pred = brain.prop.predict().detach().numpy()
        self.est_angles[step] = utils.denormalize(prop_pred, c.norm_polar)

        self.pos[step] = body.poses[:, :2]
        self.est_pos[step] = body.get_poses(self.est_angles[step],
                                            c.lengths)[:, :2]

        self.targets_pos[trial, step] = objects.obj_pos.copy()
        vis_pred = brain.vis.predict().detach().numpy()
        self.est_targets_pos[trial, step] = utils.denormalize(
            vis_pred[0, 1:], c.norm_cart)

        self.states[trial, step] = brain.discrete.prior[0]
        self.disc_obs[trial, step] = brain.discrete.qo[1]
        self.L_int[trial, step] = brain.int.L
        self.L_ext[trial, step] = brain.ext.L

    # Save log to file
    def save_log(self):
        np.savez_compressed('simulation/log_{}'.format(c.log_name),
                            pos=self.pos,
                            est_pos=self.est_pos,
                            targets_pos=self.targets_pos,
                            est_targets_pos=self.est_targets_pos,
                            states=self.states,
                            disc_obs=self.disc_obs,
                            L_int=self.L_int, L_ext=self.L_ext)
