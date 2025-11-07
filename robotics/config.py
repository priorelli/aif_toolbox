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
eta_x_int = [0, 0, 0, 0]
x_int_start = None

pi_eta_x_int = 0.0
pi_x_int = 0.5
pi_x_m_int = pi_x_int
p_x_int = 1.0
pi_prop = 1.0

pi_eta_x_ext = 0.5
pi_x_ext = pi_x_int
pi_x_m_ext = pi_x_ext
p_x_ext = p_x_int
pi_vis = 1.0

lambda_int = 0.0
lambda_ext = 0.5

lr_int = 1.0
lr_ext = 1.0
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
