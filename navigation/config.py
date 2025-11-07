# Window
width = 1000
height = 800
off_x = 0
off_y = 0

debug = 0
fps = 200

# Environment
dt = 0.3

w_s = 0
w_a = 0

t_pos = [-250, -250], [-250, 250], [250, -250], [250, 250]
t_size = 30
t_vel = 2, 2, 2, 2
t_dir = None, None, None, None

# Brain (continuous)
eta_x_pos = [0, 0]
x_pos_start = None

pi_eta_x_pos = 0.0
pi_x_pos = 1.0
pi_x_m_pos = pi_x_pos
p_x_pos = 2.0
pi_prop = 1.0
pi_vis = 1.0

lambda_pos = 0.1
lr_pos = 1.0
lr_a = 1.0

# Brain (discrete)
n_tau = 50
n_policy = 2

gain_prior = 1.0
gain_evidence = 1.0
w_bmc = 1.0

# Simulation
n_objects = 4
n_trials = 50
n_steps = 5000
n_orders = 2
log_name = ''

norm_cart = [-width, width]
