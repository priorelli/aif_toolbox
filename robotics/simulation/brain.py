import numpy as np
import torch
import utils
import config as c
from simulation.unit import DynUnit, Obs
from simulation.discrete import Discrete


# Get proprioceptive prediction
def g_prop(x):
    return x[0, 0]


# Get visual prediction
def g_vis(x):
    return x


# Get extrinsic prediction
def g_ext(x):
    lengths_norm = utils.normalize(torch.tensor(
        c.lengths, dtype=torch.float32), c.norm_cart)
    return utils.kinematics(x, lengths_norm, c.norm_polar)


# Stay
def f_0(x, lmbda):
    return (x * 0.0).unsqueeze(0)


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
        int_inputs = utils.normalize(
                            torch.tensor(c.eta_x_int), c.norm_polar)
        self.int = DynUnit(dim=(c.n_orders, c.n_objects + 1, c.n_joints),
                           inputs=int_inputs, disc_obs=self.discrete.qo[0],
                           pi_eta_x=c.pi_eta_x_int, p_x=c.p_x_int,
                           pi_x=c.pi_x_int, pi_x_m=c.pi_x_m_int,
                           lr=c.lr_int, f=f_b, lmbda=c.lambda_int)

        self.ext = DynUnit(dim=(c.n_orders, c.n_objects + 1, 2),
                           inputs=[self.int], disc_obs=self.discrete.qo[1],
                           pi_eta_x=c.pi_eta_x_ext, p_x=c.p_x_ext,
                           pi_x=c.pi_x_ext, pi_x_m=c.pi_x_m_ext,
                           lr=c.lr_ext, f=f_b, lmbda=c.lambda_ext, g=g_ext)

        self.prop = Obs(dim=c.n_joints, inputs=[self.int],
                        pi_o=c.pi_prop, g=g_prop, lr=c.lr_a)

        self.vis = Obs(dim=(c.n_orders, c.n_objects + 1, 2),
                       inputs=[self.ext], pi_o=c.pi_vis, g=g_vis)

        self.units = [self.int, self.ext]
        self.obs = [self.prop, self.vis]

    # Initialize beliefs
    def init_belief(self, angles, pos):
        int_start = angles if c.x_int_start is None else c.x_int_start
        int_start_norm = utils.normalize(np.array(int_start), c.norm_polar)
        self.int.x[0, :] = torch.tensor(int_start_norm)

        ext_start = pos if c.x_int_start is None else (
            g_ext(self.int.x[0]))[0]
        ext_start_norm = utils.normalize(np.array(ext_start), c.norm_cart)
        self.ext.x[0, :] = torch.tensor(ext_start_norm)

        self.int.x[1] = 0.0
        self.ext.x[1] = 0.0

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
