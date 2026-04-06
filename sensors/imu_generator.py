import math
import random

class IMUSimulator:
    """
    Generates synthetic, noisy IMU data based on basic kinematics.
    Approximates a torso-mounted sensor using finite differences.
    """
    def __init__(self, accel_noise_std=0.2, gyro_noise_std=0.05, gravity=9.81):
        self.accel_noise_std = accel_noise_std
        self.gyro_noise_std = gyro_noise_std
        self.gravity = gravity
        # Tracks previous state per pedestrian ID to compute derivatives
        self._history = {}

    def reset_pedestrian(self, ped_id):
        """Initializes or resets the velocity buffer for a pedestrian."""
        self._history[ped_id] = {"vx": 0.0, "vy": 0.0, "theta": 0.0}

    def remove_pedestrian(self, ped_id):
        """Cleans up memory when a pedestrian leaves the simulation."""
        self._history.pop(ped_id, None)

    def update(self, ped, dt):
        """Computes synthetic IMU readings from velocity changes."""
        hist = self._history.get(ped.id)
        if not hist:
            self.reset_pedestrian(ped.id)
            hist = self._history[ped.id]

        # Approximate linear acceleration: a = dv/dt
        ax_true = (ped.vx - hist["vx"]) / dt
        ay_true = (ped.vy - hist["vy"]) / dt

        # Determine heading (theta) to find angular rate
        speed = math.hypot(ped.vx, ped.vy)
        if speed > 0.01:
            theta = math.atan2(ped.vy, ped.vx)
        else:
            # If stationary, maintain previous heading to avoid jumping
            theta = hist["theta"]

        # Handle phase wrap-around (e.g., jumping from -pi to pi)
        d_theta = (theta - hist["theta"] + math.pi) % (2 * math.pi) - math.pi
        gz_true = d_theta / dt

        # Update state history
        hist["vx"] = ped.vx
        hist["vy"] = ped.vy
        hist["theta"] = theta

        # Add Gaussian noise
        return {
            "ax": ax_true + random.gauss(0, self.accel_noise_std),
            "ay": ay_true + random.gauss(0, self.accel_noise_std),
            "az": self.gravity + random.gauss(0, self.accel_noise_std),
            "gx": random.gauss(0, self.gyro_noise_std),
            "gy": random.gauss(0, self.gyro_noise_std),
            "gz": gz_true + random.gauss(0, self.gyro_noise_std)
        }