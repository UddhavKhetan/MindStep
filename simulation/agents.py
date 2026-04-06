import math
import random
import numpy as np
from collections import deque

class Pedestrian:
    def __init__(self, ped_id, x=-15.0, y=-10.0, ped_type="normal", group_id=None):
        self.id = ped_id
        self.group_id = group_id if group_id is not None else ped_id
        self.x = x
        self.y = y
        self.vx = 0.0
        self.vy = 0.0
        self.ped_type = ped_type
        
        # Enhanced Accessibility & Group Profiles
        if ped_type == "slow_walker":
            self.speed = 0.8
            self.reaction_delay = 0.0
        elif ped_type == "distracted":
            self.speed = 1.4
            self.reaction_delay = 2.0 
        elif ped_type == "elderly":
            self.speed = 0.9
            self.reaction_delay = 1.5
        elif ped_type == "child":
            self.speed = 1.0
            self.reaction_delay = 1.0
            # Children have erratic slight y-drifts
            self.y += random.uniform(-1.0, 1.0)
        elif ped_type == "hurried":
            self.speed = 1.6
            self.reaction_delay = -0.5 # Anticipates green
        else: # normal
            self.speed = 1.4
            self.reaction_delay = 0.0

        self.state = "APPROACHING"
        self.has_near_miss = False
        self.completed_crossing = False
        self.will_cross = True     # <-- ADDED THIS BACK IN!
        
        # ML and Uncertainty Tracking
        self.intent_prob = 0.0 
        self.intent_history = deque(maxlen=5)
        self.prediction_entropy = 0.0
        self.epistemic_variance = 0.0
        
        self.wait_time = 0.0
        self.curb_x = -5.0
        self.clear_x = 5.0

    def update_intent(self, prob):
        self.intent_prob = prob
        self.intent_history.append(prob)
        # Shannon entropy: H = -p log2(p) - (1-p) log2(1-p)
        eps = 1e-9
        p = np.clip(prob, eps, 1 - eps)
        self.prediction_entropy = - (p * np.log2(p) + (1 - p) * np.log2(1 - p))
        self.epistemic_variance = np.var(self.intent_history) if len(self.intent_history) > 1 else 0.0

    def step(self, dt, env):
        if self.state == "APPROACHING":
            self.vx = self.speed
            self.vy = 0.0
            self.x += self.vx * dt
            if self.x >= self.curb_x:
                self.x = self.curb_x
                self.vx = 0.0
                self.state = "WAITING_AT_CURB"

        elif self.state == "WAITING_AT_CURB":
            self.vx = 0.0
            self.vy = 0.0
            self.wait_time += dt
            if env.controller.is_pedestrian_cross_allowed():
                if self.reaction_delay > 0:
                    self.reaction_delay -= dt
                else:
                    self.state = "CROSSING"

        elif self.state == "CROSSING":
            self.vx = self.speed
            self.vy = 0.0
            self.x += self.vx * dt
            if self.x >= self.clear_x:
                self.state = "CLEARED"
                self.completed_crossing = True

        elif self.state == "CLEARED":
            self.vx = self.speed
            self.vy = 0.0
            self.x += self.vx * dt


class Vehicle:
    def __init__(self, veh_id, x=0.0, y=50.0, desired_speed=10.0):
        self.id = veh_id
        self.x = x
        self.y = y
        self.vx = 0.0  # Explicitly define X velocity as 0
        self.vy = desired_speed
        self.desired_speed = desired_speed  # <-- ADDED THIS BACK IN!
        self.delay_time = 0.0
        
        self.stop_line_y = 5.0
        self.crosswalk_y = -8.0

    def step(self, dt, env):
        light = env.controller.get_vehicle_light("S")
        ped_in_crosswalk = any(p.state == "CROSSING" for p in env.pedestrians)

        target_stop_y = None
        if light == "RED" and self.y > self.stop_line_y:
            target_stop_y = self.stop_line_y
        
        if ped_in_crosswalk and self.y > self.crosswalk_y:
            if target_stop_y is None or self.crosswalk_y < target_stop_y:
                target_stop_y = self.crosswalk_y

        if target_stop_y is not None and (self.y - target_stop_y) < 20.0:
            self.vy = max(0.0, self.vy - 4.0 * dt)
        else:
            self.vy = min(self.desired_speed, self.vy + 2.0 * dt)

        # Record delay if vehicle is forced to travel significantly below desired speed
        if self.vy < (self.desired_speed * 0.5):
            self.delay_time += dt

        self.y -= self.vy * dt