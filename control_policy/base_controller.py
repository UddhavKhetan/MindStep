class TrafficController:
    """
    Minimal fixed-time traffic controller.
    Cycles between NS_GREEN and EW_GREEN.
    """
    def __init__(self, phase_duration=30.0):
        self.phases = ["NS_GREEN", "EW_GREEN"]
        self.current_phase_idx = 0
        self.time_in_phase = 0.0
        self.phase_duration = phase_duration

    @property
    def current_phase(self):
        return self.phases[self.current_phase_idx]

    def step(self, dt, env=None):
        """Advances the controller time and handles phase transitions."""
        self.time_in_phase += dt
        if self.time_in_phase >= self.phase_duration:
            self.time_in_phase -= self.phase_duration
            self.current_phase_idx = (self.current_phase_idx + 1) % len(self.phases)

    def is_pedestrian_cross_allowed(self, crosswalk_id=None):
        """
        Pedestrians cross the North-South leg (walking East-West).
        They are allowed to cross when the EW traffic has the green light.
        """
        return self.current_phase == "EW_GREEN"

    def get_vehicle_light(self, direction):
        """Returns the signal state for a given vehicle approach direction."""
        if direction in ["N", "S"]:
            return "GREEN" if self.current_phase == "NS_GREEN" else "RED"
        else:
            return "GREEN" if self.current_phase == "EW_GREEN" else "RED"