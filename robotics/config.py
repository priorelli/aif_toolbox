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
eta_x_int = [0, 0, 0, 0]  # Prior of hidden states
x_int_start = None

pi_eta_x_int = 0.0  # Precision of prior prediction error (int)
pi_x_int = 0.5  # Precision of dynamics prediction error (int)
pi_x_m_int = pi_x_int  # Precision of reduced models (int)
p_x_int = 1.0
pi_prop = 1.0  # Precision of prop prediction error

pi_eta_x_ext = 0.5  # Precision of prior prediction error (ext)
pi_x_ext = pi_x_int  # Precision of dynamics prediction error (ext)
pi_x_m_ext = pi_x_ext  # Precision of reduced models (ext)
p_x_ext = p_x_int
pi_vis = 1.0  # Precision of visual prediction error

lambda_int = 0.0  # Attractor gain (int)
lambda_ext = 0.5  # Attractor gain (ext)

lr_int = 1.0  # Learning rate (int)
lr_ext = 1.0  # Learning rate (ext)
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

start = [-20, 70, 100, 20]
lengths = [100, 140, 180, 40]

# Body
joints = {}
joints['trunk'] = {'link': None, 'angle': start[0],
                   'size': (lengths[0], 50)}
joints['shoulder'] = {'link': 'trunk', 'angle': start[1],
                      'size': (lengths[1], 40)}
joints['elbow'] = {'link': 'shoulder', 'angle': start[2],
                   'size': (lengths[2], 36)}
joints['wrist'] = {'link': 'elbow', 'angle': start[3],
                   'size': (lengths[3], 36)}
n_joints = len(joints)

norm_polar = [-180.0, 180.0]
norm_cart = [-sum(lengths), sum(lengths)]
