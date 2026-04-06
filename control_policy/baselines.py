from control_policy.smart_controller import SmartController

class HeuristicPedestrianController(SmartController):
    """Extends phase if pedestrian is physically near the curb (No ML)."""
    def step(self, dt, env=None):
        self.time_in_phase += dt
        high_intent_waiting = False
        if env:
            for p in env.pedestrians:
                if p.state in ["APPROACHING", "WAITING_AT_CURB"] and abs(p.curb_x - p.x) <= 4.0:
                    high_intent_waiting = True
                    break
        # Force parent to use this heuristic bool, bypassing ML intent
        setattr(env, '_temp_high_intent', high_intent_waiting)
        super().step(dt, None) # Call super without env to prevent it recalculating ML intent
        if getattr(env, '_temp_high_intent'):
            # Re-inject the logic for extensions since super didn't have env
            if self.current_phase == "EW_GREEN" and not self._is_extending and self.time_in_phase >= (self.nominal_durations["EW_GREEN"] - self.grace_window):
                self._is_extending = True
                self.extensions_count += 1

class PushButtonController(SmartController):
    """Old-school button: Only triggers if pedestrian is explicitly STOPPED at curb."""
    def step(self, dt, env=None):
        self.time_in_phase += dt
        button_pressed = False
        if env:
            for p in env.pedestrians:
                if p.state == "WAITING_AT_CURB":
                    button_pressed = True
                    break
        setattr(env, '_temp_high_intent', button_pressed)
        super().step(dt, None)