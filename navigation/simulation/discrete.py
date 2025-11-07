import numpy as np
import torch
import jax.numpy as jnp
from jax.nn import softmax
from pymdp import utils as pymdputils
from pymdp.agent import Agent
from pymdp import control
import utils
import config as c


# Define discrete class
class Discrete:
    def __init__(self):
        # Hidden states
        self.pos_states = ['T1', 'T2', 'T3', 'T4']
        self.n_states = [len(self.pos_states)]
        self.n_obs = [len(self.pos_states)]

        # Likelihood matrix
        self.A, self.A_deps = self.get_A()

        # Transition matrix
        self.actions = ['TO_T1', 'TO_T2', 'TO_T3', 'TO_T4']
        self.n_controls = [len(self.actions) for _ in range(len(self.n_states))]
        self.B, self.B_deps = self.get_B()

        # Preference matrix
        self.C = pymdputils.list_array_zeros(self.n_obs)
        self.C[0] = softmax(jnp.array([0.0, 1.0, 0.0, 0.0]))

        # Prior matrix
        self.D = pymdputils.list_array_zeros(self.n_states)
        self.D[0] = jnp.ones(self.n_states[0]) / self.n_states[0]

        # Initialize agent
        self.agent = Agent(self.A, self.B, self.C, self.D,
                           A_dependencies=self.A_deps,
                           B_dependencies=self.B_deps,
                           policy_len=c.n_policy, onehot_obs=True,
                           learn_A=False, learn_B=False, learn_D=False)

        # Set prior and expected obs
        self.prior = self.agent.D
        self.qo = [torch.from_numpy(np.array(qo))
                   for qo in self.get_expected_obs()]

    # Get likelihood matrix
    def get_A(self):
        A_deps = [[0]]
        A_m_shapes = [[self.n_obs[o]] + [self.n_states[s] for s in A_deps[o]]
                      for o in range(len(self.n_obs))]
        A = [np.zeros(shape, dtype=np.float32) for shape in A_m_shapes]

        A[0] = np.eye(self.n_obs[0])

        A = [jnp.array(a) for a in A]
        return A, A_deps

    # Get transition matrix
    def get_B(self):
        B_deps = [[0]]
        B_f_shapes = [[self.n_states[a]] + [self.n_states[s] for s in B_deps[a]] +
                      [self.n_controls[a]] for a in range(len(self.n_controls))]
        B = [np.zeros(shape, dtype=np.float32) for shape in B_f_shapes]

        for action_id, action_label in enumerate(self.actions):
            for idx, state in enumerate(self.pos_states):
                new_idx = idx

                if action_label == 'TO_T1':
                    new_idx = 0
                elif action_label == 'TO_T2':
                    new_idx = 1
                if action_label == 'TO_T3':
                    new_idx = 2
                if action_label == 'TO_T4':
                    new_idx = 3

                B[0][new_idx, idx, action_id] = 1.0

        B = [jnp.array(b) for b in B]
        return B, B_deps

    # Compute expected observations
    def get_expected_obs(self):
        qs = [x.squeeze() if x.ndim == 2 and x.shape[0] == 1 else
              x for x in self.prior]
        A = [x.squeeze() for x in self.agent.A]
        return control.compute_expected_obs(qs, A, self.A_deps)

    # Run discrete step
    def step(self, L):
        # Perform BMC
        qr = jnp.array(utils.bmc(self.qo[0], L[0], c.w_bmc,
                       c.gain_prior, c.gain_evidence).detach().numpy())
        qr = [jnp.reshape(qr, (1, 1, len(qr)))]

        # Infer new state
        qs = self.agent.infer_states(observations=qr,
                                     empirical_prior=self.prior)

        # Infer policies
        qpi, G = self.agent.infer_policies(qs)
        # for i, g in enumerate(G[0]):
        #     print(f'Policy {i}: G = {g:.4f}')

        # Sample action
        action = self.agent.sample_action(qpi)

        # Learn parameters
        # self.agent = self.agent.infer_parameters(qs, o, action, lr_pA=1.0)

        # Update prior and expected obs
        self.prior, _ = self.agent.update_empirical_prior(action, qs)
        for i, qo in enumerate(self.get_expected_obs()):
            self.qo[i][:] = torch.from_numpy(np.array(qo))

        if c.debug:
            print()
            print('qr: ', qr)
            print('qs: ', qs)
            print('qs_next: ', self.prior)
            print('qo_next: ', self.qo)
            print()
