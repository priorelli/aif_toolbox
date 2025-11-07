import numpy as np
import torch
import utils
import config as c
from simulation.unit import DynUnit, Obs
from simulation.discrete import Discrete


# Get proprioceptive prediction
def g_prop(x):
    return x[:, 0]


# Get visual prediction
def g_vis(x):
    return x[:, 1:]


# Reach balls
def f_b(x, lmbda):
    a = x[1:].unsqueeze(1).expand(-1, c.n_objects, -1)
    b = x[1:].unsqueeze(0).expand(c.n_objects, -1, -1)

    x_stacked = torch.cat([a[:, :1, :], b], dim=1)

    return (x_stacked - x.unsqueeze(0)) * lmbda


# Define brain class
class Brain:
    def __init__(self):
        # Initialize discrete
        self.discrete = Discrete()

        # Initialize units
        pos_inputs = np.tile(utils.normalize(
            np.array(c.eta_x_pos), c.norm_cart), (c.n_objects + 1, 1))
        self.pos = DynUnit(dim=(c.n_orders, c.n_objects + 1, 2),
                           inputs=pos_inputs, disc_obs=self.discrete.qo[0],
                           pi_eta_x=c.pi_eta_x_pos, p_x=c.p_x_pos,
                           pi_x=c.pi_x_pos, pi_x_m=c.pi_x_m_pos,
                           lr=c.lr_pos, f=f_b, lmbda=c.lambda_pos)

        self.prop = Obs(dim=(c.n_orders, 2), inputs=[self.pos],
                        pi_o=c.pi_prop, g=g_prop, lr=c.lr_a)

        self.vis = Obs(dim=(c.n_orders, c.n_objects, 2), inputs=[self.pos],
                       pi_o=c.pi_vis, g=g_vis)

        self.units = [self.pos]
        self.obs = [self.prop, self.vis]

    # Initialize beliefs
    def init_belief(self, pos, obj_pos):
        pos_start = pos if c.x_pos_start is None \
            else c.x_pos_start
        pos_start_norm = utils.normalize(pos_start, c.norm_cart)
        self.pos.x[0, :] = torch.tensor(pos_start_norm)

        # obj_start_norm = utils.normalize(obj_pos, c.norm_cart)
        # self.pos.x[0, 1:] = torch.tensor(obj_start_norm)

        self.pos.x[1] = 0.0
        self.prop.o[:] = 0.0
        self.vis.o[:] = 0.0
        self.prop.actions[:] = 0.0

    # Run an inference step
    def inference_step(self, O, step):
        # Run discrete step
        if (step + 1) % c.n_tau == 0:
            self.discrete.step([unit.L for unit in self.units])
            for unit in self.units:
                unit.L[:] = 0

        # Set observations
        self.prop.o = torch.tensor(O[0])
        self.vis.o = torch.tensor(O[1])

        # Perform message passing step
        for unit in self.units + self.obs:
            unit.step()

        # Update all units
        for unit in self.units + self.obs:
            unit.update(c.dt)

        return utils.denormalize(self.prop.actions, c.norm_cart)
