from control_policy.smart_controller import SmartController

class HeuristicPedestrianController(SmartController):
    """
    Rule-based baseline. Extends phase solely on spatial presence, ignoring ML probabilities.
    """
    def step(self, dt, env=None):
        self.time_in_phase += dt
        current_nominal = self.nominal_durations[self.current_phase]

        # Ignore ML, just use rules
        high_intent_waiting = False
        if env:
            for p in env.pedestrians:
                if p.state in ["APPROACHING", "WAITING_AT_CURB"]:
                    dist_to_curb = abs(p.curb_x - p.x)
                    if dist_to_curb <= self.curb_distance_threshold:
                        high_intent_waiting = True
                        break

        if high_intent_waiting:
            self.high_intent_events += 1

        # Identical phase transition logic as SmartController
        if self.current_phase == "EW_GREEN":
            in_decision_window = (current_nominal - self.grace_window) <= self.time_in_phase <= current_nominal
            if in_decision_window and not self._is_extending and not self._opportunity_logged:
                self.extension_opportunities += 1
                self._opportunity_logged = True
            
            if in_decision_window and high_intent_waiting and not self._is_extending:
                self._is_extending = True
                self.extensions_count += 1
            
            target_duration = current_nominal + self.max_extension if self._is_extending else current_nominal
            if self.time_in_phase >= target_duration:
                self._transition(target_duration)
            elif self._is_extending and self.time_in_phase >= current_nominal:
                self.total_extension_time += dt

        elif self.current_phase == "NS_GREEN":
            min_green = 5.0
            vehicle_in_dilemma = any(-12.0 <= v.y <= 20.0 for v in env.vehicles) if env else False
            
            if self.time_in_phase >= current_nominal and not vehicle_in_dilemma:
                self._transition(self.time_in_phase)
            elif high_intent_waiting and self.time_in_phase >= min_green and not vehicle_in_dilemma:
                self._transition(self.time_in_phase)