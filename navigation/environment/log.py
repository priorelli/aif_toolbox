import numpy as np
import utils
import config as c


# Define log class
class Log:
    def __init__(self):
        # Initialize logs
        self.pos = np.zeros((c.n_trials, c.n_steps, 2))
        self.est_pos = np.zeros_like(self.pos)

        self.targets_pos = np.zeros((c.n_trials, c.n_steps, c.n_objects, 2))
        self.est_targets_pos = np.zeros_like(self.targets_pos)

        self.states = np.zeros((c.n_trials, c.n_steps, c.n_objects))
        self.disc_obs = np.zeros((c.n_trials, c.n_steps, c.n_objects))

        self.L = np.zeros((c.n_trials, c.n_steps, c.n_objects))

    # Track logs for each iteration
    def track(self, trial, step, brain, body, objects):
        self.pos[trial, step] = body.pos.copy()
        prop_pred = brain.prop.predict().detach().numpy()
        self.est_pos[trial, step] = utils.denormalize(
            prop_pred[0], c.norm_cart)

        self.targets_pos[trial, step] = objects.obj_pos.copy()
        vis_pred = brain.vis.predict().detach().numpy()
        self.est_targets_pos[trial, step] = utils.denormalize(
            vis_pred[0], c.norm_cart)

        self.states[trial, step] = brain.discrete.prior[0]
        self.disc_obs[trial, step] = brain.discrete.qo[0]
        self.L[trial, step] = brain.pos.L

    # Save log to file
    def save_log(self):
        np.savez_compressed('simulation/log_{}'.format(c.log_name),
                            pos=self.pos,
                            est_pos=self.est_pos,
                            targets_pos=self.targets_pos,
                            est_targets_pos=self.est_targets_pos,
                            states=self.states,
                            disc_obs=self.disc_obs,
                            L=self.L)
