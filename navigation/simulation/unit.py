import numpy as np
import torch
import utils


# Define dynamic unit class
class DynUnit:
    def __init__(self, dim, inputs, disc_obs, pi_eta_x, p_x,
                 pi_x, pi_x_m, lr, f, lmbda, g=None):
        # Point to inputs
        self.inputs = inputs
        self.disc_obs = disc_obs
        self.L = torch.zeros(len(disc_obs))

        # Initialize hidden states
        self.x = torch.zeros(dim)

        # Initialize precisions
        self.pi_eta_x = torch.tensor(pi_eta_x)
        self.p_x = torch.tensor(p_x)
        self.pi_x = torch.tensor(pi_x)
        self.pi_x_m = torch.tensor(pi_x_m)

        # Initialize likelihood and dynamics
        self.g = g
        self.f = f

        # Initialize learning rate
        self.lr = lr
        self.lmbda = lmbda

        # Initialize dynamics
        self.pred_x = torch.zeros_like(self.x[1])
        self.Preds_x = torch.zeros((len(self.disc_obs), len(self.x[1])))

        # Initialize prediction errors and gradients
        self.eps_eta_x = torch.zeros_like(self.x[0])
        self.eps_x = torch.zeros_like(self.x[1])

        self.grad_o = torch.zeros_like(self.x)
        self.grad_x = torch.zeros_like(self.x[0])

    # Perform likelihood step
    def step_likelihood(self):
        if self.g is None:
            self.eps_eta_x = (self.x[0] - self.inputs) * self.pi_eta_x
        else:
            # Get priors from parent units
            Eta_x = [inpt.x[0].clone().detach().requires_grad_(True)
                     for inpt in self.inputs]

            # Predict hidden states
            pred_eta_x = self.g(*Eta_x)

            # Compute prior prediction error
            x0 = self.x[0].clone().detach()
            self.eps_eta_x = (x0 - pred_eta_x) * self.pi_eta_x

            # Backpropagate gradient to parent units
            self.eps_eta_x.backward(self.eps_eta_x)
            for eta_x, inpt in zip(Eta_x, self.inputs):
                inpt.grad_o += eta_x.grad

    # Perform dynamics step
    def step_dynamics(self):
        # Get parents
        x = self.x[0].clone().detach().requires_grad_(True)

        # Predict dynamics
        self.Preds_x = self.f(x, self.lmbda)
        pred_x = torch.tensordot(self.disc_obs, self.Preds_x, dims=([0], [0]))

        # Compute dynamics prediction error
        x1 = self.x[1].clone().detach()
        self.eps_x = (x1 - pred_x) * self.pi_x

        # Backpropagate gradients to parents
        self.eps_x.backward(self.eps_x)
        self.grad_x = x.grad

        # Accumulate log evidence
        self.L += utils.acc_log_evidence(pred_x, self.Preds_x, x1,
                                         self.pi_x, self.pi_x_m, self.p_x)

    # Perform a step
    def step(self):
        self.step_likelihood()
        self.step_dynamics()

    # Integrate with gradient descent
    def update(self, dt):
        # Compute free energy derivative
        dF_dx = self.grad_o
        dF_dx[0] += self.eps_eta_x + self.grad_x
        dF_dx[1] += self.eps_x

        # Compute belief derivative
        x_dot = utils.shift(self.x) - dF_dx.clone().detach()

        # Update belief
        self.x += dt * x_dot * self.lr

        # Clear variables
        self.eps_eta_x = torch.zeros_like(self.x[0])
        self.grad_o = torch.zeros_like(self.x)


# Define static unit class
class StaticUnit:
    def __init__(self, dim, disc_obs, pi_eta_x, pi_eta_x_m,
                 p_eta_x, pi_x, lr, f, lmbda, Eta_m):
        # Point to discrete observations
        self.disc_obs = disc_obs
        self.L = torch.zeros_like(disc_obs)

        # Initialize hidden states
        self.x = torch.zeros(dim)

        # Initialize precisions
        self.pi_eta_x = torch.array(pi_eta_x)
        self.pi_eta_x_m = torch.array(pi_eta_x_m)
        self.p_eta_x = torch.array(p_eta_x)
        self.pi_x = torch.array(pi_x)

        # Initialize priors of reduced models and dynamics
        self.Eta_m = Eta_m
        self.f = f

        # Initialize learning rate
        self.lr = lr
        self.lmbda = lmbda

        # Initialize prediction errors and gradients
        self.eps_eta_x = torch.zeros_like(self.x[0])
        self.eps_x = torch.zeros_like(self.x[1])

        self.grad_o = torch.zeros_like(self.x)
        self.grad_x = torch.zeros_like(self.x[0])

    # Perform likelihood step
    def step_likelihood(self):
        # Bayesian model average
        eta_x = torch.dot(self.disc_obs, self.Eta_m)

        # Compute prior prediction error
        self.eps_eta_x = (self.x[0] - eta_x) * self.pi_eta_x

        # Accumulate log evidence
        self.L += utils.acc_log_evidence(
            eta_x, self.Eta_m, self.x[0], self.pi_eta_x,
            self.pi_eta_x_m, self.p_eta_x)

    # Perform dynamics step
    def step_dynamics(self):
        # Get parents
        x = self.x[0].clone().detach().requires_grad_(True)

        # Predict dynamics
        pred_x = self.f(x, self.lmbda)

        # Compute dynamics prediction error
        x1 = self.x[1].clone().detach()
        self.eps_x = (x1 - pred_x) * self.pi_x

        # Backpropagate gradients to parents
        self.eps_x.backward(self.eps_x)
        self.grad_x = x.grad

    # Perform a step
    def step(self):
        self.step_likelihood()
        self.step_dynamics()

    # Integrate with gradient descent
    def update(self, dt):
        # Compute free energy derivative
        dF_dx = self.grad_o
        dF_dx[0] += self.eps_eta_x + self.grad_x
        dF_dx[1] += self.eps_x

        # Compute belief derivative
        x_dot = utils.shift(self.x) - dF_dx.clone().detach()

        # Update belief
        self.x += dt * x_dot * self.lr

        # Clear variables
        self.eps_eta_x = torch.zeros_like(self.x[0])
        self.grad_o = torch.zeros_like(self.x)


# Define observation class
class Obs:
    def __init__(self, dim, inputs, pi_o, g, lr=None):
        # Point to inputs
        self.inputs = inputs

        # Initialize observation and actions
        self.o = torch.zeros(dim)
        self.actions = np.zeros_like(self.o[0]) if lr else None

        # Initialize precision
        self.pi_o = torch.tensor(pi_o)

        # Initialize likelihood
        self.g = g

        # Initialize learning rate
        self.lr = lr

        # Initialize prediction error
        self.eps_o = torch.zeros_like(self.o)

    # Compute prediction
    def predict(self, get_x=False):
        # Get priors from parent units
        X = [inpt.x.clone().detach().requires_grad_(get_x)
             for inpt in self.inputs]

        # Compute prediction
        if not get_x:
            return self.g(*X)
        return X, self.g(*X)

    # Perform a step
    def step(self):
        # Predict hidden state
        X, pred_o = self.predict(get_x=True)

        # Compute prior prediction error
        self.eps_o = (self.o - pred_o) * self.pi_o

        # Backpropagate gradient to parent units
        self.eps_o.backward(self.eps_o)
        for eta_o, inpt in zip(X, self.inputs):
            inpt.grad_o += eta_o.grad

    # Integrate with gradient descent
    def update(self, dt):
        if self.actions is not None:
            # Compute free energy derivative
            a_dot = -dt * self.eps_o[0].detach().numpy()

            # Update actions
            self.actions += dt * a_dot * self.lr
