import math
import random
import numpy as np
import pandas as pd
from collections import deque
import scipy.stats

from control_policy.base_controller import TrafficController
from simulation.agents import Pedestrian, Vehicle
from sensors.imu_generator import IMUSimulator
from ml_intention.model import IntentionPredictor
from ml_intention.features import extract_features_from_window

class IntersectionEnv:
    def __init__(self, controller=None, seed=None, scenario="normal"):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.time = 0.0
        self.pedestrians = []
        self.vehicles = []
        self.near_misses = 0
        self.scenario = scenario

        # Global metrics for statistical arrays
        self.ped_wait_times = []
        self.veh_delay_times = []
        self.ped_type_waits = {
            "normal": [], "slow_walker": [], "distracted": [], 
            "elderly": [], "child": [], "hurried": [], "paired_group": []
        }
        self.severe_conflicts = 0
        self.ttc_records = []
        self.min_ttc_observed = float('inf')
        
        self.controller = controller if controller else TrafficController(phase_duration=15.0)

        self.imu_simulator = IMUSimulator()
        self.imu_log = []
        self.window_size = 10
        self.imu_buffers = {} 
        self.intent_log = []
        
        try:
            self.intention_model = IntentionPredictor.load("intention_model.joblib")
        except FileNotFoundError:
            self.intention_model = None

        self._next_ped_id = 1
        self._next_veh_id = 1
        self._time_since_last_ped = 0.0
        self._time_since_last_veh = 0.0

        self.cw_x_min, self.cw_x_max = -5.0, 5.0
        self.cw_y_min, self.cw_y_max = -12.0, -8.0

        # Scenario Configuration
        self.spawn_rates = {
            "normal": {"ped": (10.0, 14.0), "veh": (6.0, 10.0)},
            "heavy_traffic": {"ped": (12.0, 16.0), "veh": (2.0, 5.0)},
            "ped_bursts": {"ped": (2.0, 6.0), "veh": (8.0, 12.0)},
        }
        self.step_counter = 0

    def spawn_pedestrian(self):
        # Determine if spawning a paired group
        if random.random() < 0.15: # 15% chance of a pair
            group_id = self._next_ped_id
            p1 = Pedestrian(ped_id=self._next_ped_id, ped_type="paired_group", group_id=group_id)
            self._next_ped_id += 1
            p2 = Pedestrian(ped_id=self._next_ped_id, y=-10.5, ped_type="paired_group", group_id=group_id)
            self.pedestrians.extend([p1, p2])
            for p in [p1, p2]:
                self.imu_simulator.reset_pedestrian(p.id)
                self.imu_buffers[p.id] = deque(maxlen=self.window_size)
            self._next_ped_id += 1
        else:
            ptype = random.choices(
                ["normal", "slow_walker", "distracted", "elderly", "child", "hurried"], 
                weights=[0.5, 0.1, 0.1, 0.15, 0.05, 0.1]
            )[0]
            ped = Pedestrian(ped_id=self._next_ped_id, ped_type=ptype)
            self.pedestrians.append(ped)
            self.imu_simulator.reset_pedestrian(ped.id)
            self.imu_buffers[ped.id] = deque(maxlen=self.window_size)
            self._next_ped_id += 1

    def spawn_vehicle(self):
        veh = Vehicle(veh_id=self._next_veh_id)
        self.vehicles.append(veh)
        self._next_veh_id += 1

    def _handle_spawning(self, dt):
        self._time_since_last_ped += dt
        self._time_since_last_veh += dt

        rates = self.spawn_rates.get(self.scenario, self.spawn_rates["normal"])
        
        if self._time_since_last_ped >= random.uniform(*rates["ped"]):
            self.spawn_pedestrian()
            self._time_since_last_ped = 0.0

        if self._time_since_last_veh >= random.uniform(*rates["veh"]):
            self.spawn_vehicle()
            self._time_since_last_veh = 0.0

    def step(self, dt):
        self.time += dt
        self.step_counter += 1 # Increment counter
        
        self._handle_spawning(dt)
        self.controller.step(dt, self)

        for p in self.pedestrians: p.step(dt, self)
        for v in self.vehicles: v.step(dt, self)

        # OPTIMIZATION: Only run heavy ML inference every 5 steps (0.5 seconds)
        predict_this_tick = (self.step_counter % 5 == 0)

        # ML Integration
        for p in self.pedestrians:
            dist_to_curb = abs(p.curb_x - p.x)
            intent_label = 1 if (p.will_cross and p.state in ["APPROACHING", "WAITING_AT_CURB"] and dist_to_curb <= 4.0) else 0
            
            imu_data = self.imu_simulator.update(p, dt)
            buffer_record = imu_data.copy()
            buffer_record["intent"] = intent_label
            self.imu_buffers[p.id].append(buffer_record)
            
            # OPTIMIZATION: Only predict if they are waiting/approaching, AND it's the right tick
            if p.state in ["APPROACHING", "WAITING_AT_CURB"]:
                if predict_this_tick and self.intention_model and len(self.imu_buffers[p.id]) == self.window_size:
                    window_df = pd.DataFrame(self.imu_buffers[p.id])
                    features = extract_features_from_window(window_df)
                    prob = self.intention_model.predict_proba(np.array([features]))[0]
                    p.update_intent(float(prob))

        # Cleanup and metrics
        active_pedestrians = []
        for p in self.pedestrians:
            if p.x < 20.0:
                active_pedestrians.append(p)
            else:
                self.ped_wait_times.append(p.wait_time)
                self.ped_type_waits[p.ped_type].append(p.wait_time)
                self.imu_simulator.remove_pedestrian(p.id)
                self.imu_buffers.pop(p.id, None)
        self.pedestrians = active_pedestrians

        active_vehicles = []
        for v in self.vehicles:
            if v.y > -50.0:
                active_vehicles.append(v)
            else:
                self.veh_delay_times.append(v.delay_time)
        self.vehicles = active_vehicles

        self._check_near_misses()

    def _check_near_misses(self):
        for p in self.pedestrians:
            if p.state == "CROSSING":
                for v in self.vehicles:
                    dist = math.hypot(p.x - v.x, p.y - v.y)
                    
                    # Near miss (Binary)
                    if dist < 3.0 and not p.has_near_miss:
                        self.near_misses += 1
                        p.has_near_miss = True
                    
                    # Time To Collision (TTC)
                    # p moving +x (p.vx), v moving -y (-v.vy)
                    closing_speed_x = p.vx - v.vx 
                    closing_speed_y = p.vy + v.vy 
                    closing_speed = math.hypot(closing_speed_x, closing_speed_y)
                    
                    # Dot product of position vector and relative velocity to see if converging
                    pos_dot_vel = (p.x - v.x)*closing_speed_x + (p.y - v.y)*(-closing_speed_y)
                    
                    if pos_dot_vel < 0 and closing_speed > 0.1: # Converging
                        ttc = dist / closing_speed
                        self.min_ttc_observed = min(self.min_ttc_observed, ttc)
                        self.ttc_records.append(ttc)
                        if ttc < 1.5:
                            self.severe_conflicts += 1