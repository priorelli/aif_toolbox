# Window
width = 1000
height = 800
off_x = 0
off_y = 0

debug = 0
fps = 200

# Environment
dt = 0.3

w_s = 0  # Sensory noise
w_a = 0  # Action noise

t_pos = [-250, -250], [-250, 250], [250, -250], [250, 250]  # Target positions
t_size = 30
t_vel = 2, 2, 2, 2  # Target velocities
t_dir = None, None, None, None  # Target directions

# Brain (continuous)
eta_x_pos = [0, 0]  # Prior of hidden states
x_pos_start = None

pi_eta_x_pos = 0.0  # Precision of prior prediction error
pi_x_pos = 1.0  # Precision of dynamics prediction error
pi_x_m_pos = pi_x_pos  # Precision of reduced models
p_x_pos = 2.0
pi_prop = 1.0  # Precision of prop prediction error
pi_vis = 1.0  # Precision of visual prediction error

lambda_pos = 0.1  # Attractor gain
lr_pos = 1.0  # Learning rate of hidden states
lr_a = 1.0

# Brain (discrete)
n_tau = 50  # Steps of evidence accumulation
n_policy = 2  # Number of discrete policies

gain_prior = 1.0  # Gain of prior in BMC
gain_evidence = 1.0  # Gain of evidence in BMC
w_bmc = 1.0  # Precision of BMC

# Simulation
n_objects = 4
n_trials = 50
n_steps = 5000
n_orders = 2
log_name = ''

norm_cart = [-width, width]
