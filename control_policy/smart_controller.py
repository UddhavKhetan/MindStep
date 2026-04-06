from control_policy.base_controller import TrafficController
import numpy as np

class SmartController(TrafficController):
    def __init__(self, ns_green_nominal=15.0, ew_green_nominal=15.0,
                 max_extension=10.0, intent_threshold=0.5, grace_window=3.0,
                 ablate_spatial=False, uncertainty_penalty_weight=0.3):
        super().__init__(phase_duration=ns_green_nominal)
        self.nominal_durations = {"NS_GREEN": ns_green_nominal, "EW_GREEN": ew_green_nominal}
        self.max_extension = max_extension
        self.intent_threshold = intent_threshold
        self.grace_window = grace_window
        self.ablate_spatial = ablate_spatial
        self.uncertainty_penalty_weight = uncertainty_penalty_weight

        self.extensions_count = 0
        self.total_extension_time = 0.0
        self._is_extending = False

    def get_intent_score(self, ped):
        """Uncertainty-propagated scoring logic."""
        prob = getattr(ped, 'intent_prob', 0.0)
        entropy = getattr(ped, 'prediction_entropy', 1.0) # Max entropy default
        
        if self.ablate_spatial:
            distance_factor = 1.0
        else:
            dist_to_curb = abs(ped.curb_x - ped.x)
            distance_factor = max(0.0, 1.0 - (dist_to_curb / 4.0))
            
        history = getattr(ped, 'intent_history', [])
        recent_high_intent_ratio = sum(1 for p in history if p > 0.6) / len(history) if history else 0
        history_boost = 1.0 + (recent_high_intent_ratio * 0.2)
        
        uncertainty_penalty = 1.0 - (self.uncertainty_penalty_weight * entropy)
        
        score = prob * distance_factor * history_boost * max(0.1, uncertainty_penalty)
        return score

    def step(self, dt, env=None):
        self.time_in_phase += dt
        current_nominal = self.nominal_durations[self.current_phase]

        high_intent_waiting = False
        if env:
            for p in env.pedestrians:
                if p.state in ["APPROACHING", "WAITING_AT_CURB"]:
                    score = self.get_intent_score(p)
                    if score >= self.intent_threshold:
                        high_intent_waiting = True
                        break

        # [Phase transition logic remains identical, using the max-out rule previously implemented]
        if self.current_phase == "EW_GREEN":
            in_decision_window = (current_nominal - self.grace_window) <= self.time_in_phase <= current_nominal
            if in_decision_window and high_intent_waiting and not self._is_extending:
                self._is_extending = True
                self.extensions_count += 1
            
            target = current_nominal + self.max_extension if self._is_extending else current_nominal
            if self.time_in_phase >= target:
                self._transition(target)
            elif self._is_extending and self.time_in_phase >= current_nominal:
                self.total_extension_time += dt

        elif self.current_phase == "NS_GREEN":
            vehicle_in_dilemma = any(-12.0 <= v.y <= 20.0 for v in env.vehicles) if env else False
            absolute_max_green = current_nominal * 2.0
            
            if self.time_in_phase >= absolute_max_green:
                self._transition(absolute_max_green)
            elif self.time_in_phase >= current_nominal and not vehicle_in_dilemma:
                self._transition(self.time_in_phase)
            elif high_intent_waiting and self.time_in_phase >= 5.0 and not vehicle_in_dilemma:
                self._transition(self.time_in_phase)

    def _transition(self, time_to_subtract: float):
        self.time_in_phase -= time_to_subtract
        self.current_phase_idx = (self.current_phase_idx + 1) % len(self.phases)
        self._is_extending = False