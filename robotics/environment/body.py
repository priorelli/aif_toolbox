import numpy as np
import config as c


# Define body class
class Body:
    def __init__(self):
        # Initialize body parameters
        ids = {joint: j for j, joint in enumerate(c.joints)}

        self.angles = np.zeros(c.n_joints)
        self.ang_vel = np.zeros(c.n_joints)
        self.size = np.zeros((c.n_joints, 2))

        self.idxs = {}

        for joint in c.joints:
            self.angles[ids[joint]] = c.joints[joint]['angle']
            self.size[ids[joint]] = c.joints[joint]['size']

            if c.joints[joint]['link'] is not None:
                self.idxs[ids[joint]] = ids[c.joints[joint]['link']]
            else:
                self.idxs[ids[joint]] = -1

        # Initialize positions
        self.poses = self.get_poses()

    # Compute pose of every link
    def get_poses(self, angles=None, lengths=None, poses=None):
        new_poses = np.zeros((c.n_joints + 1, 3))

        if angles is None:
            angles = self.angles
        if lengths is None:
            lengths = self.size[:, 0]
        if poses is None:
            poses = new_poses

        for j in range(c.n_joints):
            old_pose = poses[self.idxs[j] + 1]
            new_poses[j + 1] = self.forward(angles[j], lengths[j], old_pose)

        return new_poses

    # Kinematic forward pass
    def forward(self, theta, length, pose):
        position, phi = pose[:2], pose[2]
        new_phi = theta + phi

        direction = np.array([np.cos(np.radians(new_phi)),
                              np.sin(np.radians(new_phi))])
        new_position = position + length * direction

        return *new_position, new_phi

    # Kinematic inverse pass
    def inverse(self, phi, length):
        return np.array([-length * np.sin(np.radians(phi)),
                         length * np.cos(np.radians(phi)), 1])

    # Update body with velocity
    def update(self, vel):
        self.ang_vel = np.array(vel)
        self.angles = self.angles + c.dt * self.ang_vel
        self.poses = self.get_poses()

    # Set rotation of every joint
    def set_rotation(self, angles):
        self.angles = np.array(angles)
        self.poses = self.get_poses()

    # Get joint velocities
    def get_vel(self):
        cart_vels = np.zeros((c.n_joints, 2))

        for j in range(c.n_joints):
            v = np.zeros(2)
            current = j
            while current != -1:
                # Contribution of current joint's angular velocity
                phi = self.angles[current]
                length = self.size[current, 0]
                vel = self.ang_vel[current] * c.dt

                # Perpendicular vector for 2D rotation
                v += vel * length * np.array([-np.sin(np.radians(phi)),
                                              np.cos(np.radians(phi))])
                current = self.idxs[current]

            cart_vels[j] = v

        return cart_vels * c.dt
