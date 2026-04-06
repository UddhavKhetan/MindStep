import pygame
import sys
import math
from collections import deque
from simulation.environment import IntersectionEnv
from control_policy.smart_controller import SmartController
from control_policy.base_controller import TrafficController
from control_policy.baselines import HeuristicPedestrianController, PushButtonController

# Presentation Constants
WIDTH, HEIGHT = 1200, 800
SCALE = 15
OX, OY = WIDTH // 2 - 150, HEIGHT // 2

# Palette
BG_COLOR = (24, 26, 31)
ROAD_COLOR = (43, 47, 56)
SIDEWALK_COLOR = (54, 58, 69)
LINE_COLOR = (200, 200, 200)
GREEN = (46, 204, 113)
RED = (231, 76, 60)
YELLOW = (241, 196, 15)
WHITE = (236, 240, 241)
HUD_BG = (30, 34, 40, 230)

PED_COLORS = {
    "normal": (52, 152, 219),
    "slow_walker": (155, 89, 182),
    "distracted": (241, 196, 15),
    "elderly": (230, 126, 34),
    "child": (255, 105, 180),
    "hurried": (26, 188, 156),
    "paired_group": (149, 165, 166)
}

def world_to_screen(x, y):
    return int(OX + x * SCALE), int(OY + y * SCALE)

class Visualizer:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Smart Crosswalk Live Simulator")
        self.font = pygame.font.SysFont("Consolas", 14)
        self.title_font = pygame.font.SysFont("Consolas", 18, bold=True)
        self.clock = pygame.time.Clock()
        
        self.controllers = [
            SmartController(intent_threshold=0.4, grace_window=1.0, max_extension=12.5), 
            HeuristicPedestrianController(), 
            TrafficController()
        ]
        self.ctrl_idx = 0
        self.env = IntersectionEnv(controller=self.controllers[self.ctrl_idx], seed=42)
        
        self.dt = 0.1
        self.sim_speed = 30
        self.paused = False
        self.show_trails = True
        
        self.selected_agent = None
        self.event_log = deque(maxlen=6)
        self.last_phase = self.env.controller.current_phase
        self.last_ext_count = 0
        
        self.trails = {} # id -> list of (x,y)
        
    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE: self.paused = not self.paused
                if event.key == pygame.K_UP: self.sim_speed = min(120, self.sim_speed + 15)
                if event.key == pygame.K_DOWN: self.sim_speed = max(10, self.sim_speed - 15)
                if event.key == pygame.K_t: self.show_trails = not self.show_trails
                if event.key == pygame.K_r: 
                    self.env = IntersectionEnv(controller=self.controllers[self.ctrl_idx], seed=42)
                    self.event_log.clear()
                    self.trails.clear()
                    self.selected_agent = None
                if event.key == pygame.K_c:
                    self.ctrl_idx = (self.ctrl_idx + 1) % len(self.controllers)
                    self.env = IntersectionEnv(controller=self.controllers[self.ctrl_idx], seed=42)
                    self.event_log.clear()
                    self.trails.clear()
                    self.selected_agent = None
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    self.select_agent(event.pos)
        return True

    def select_agent(self, mouse_pos):
        mx, my = mouse_pos
        closest = None
        min_dist = 20 # Pixel snap threshold
        
        for p in self.env.pedestrians:
            sx, sy = world_to_screen(p.x, p.y)
            dist = math.hypot(mx - sx, my - sy)
            if dist < min_dist:
                min_dist = dist
                closest = ("ped", p)
                
        for v in self.env.vehicles:
            sx, sy = world_to_screen(v.x, v.y)
            dist = math.hypot(mx - sx, my - sy)
            if dist < min_dist + 10: # Vehicles have larger hitboxes
                min_dist = dist
                closest = ("veh", v)
                
        self.selected_agent = closest

    def update_logic(self):
        if not self.paused:
            self.env.step(self.dt)
            
            # Event Logging Logic
            curr_phase = self.env.controller.current_phase
            curr_ext = getattr(self.env.controller, 'extensions_count', 0)
            
            if curr_phase != self.last_phase:
                self.event_log.appendleft(f"[{self.env.time:.1f}s] Phase switch: {curr_phase}")
                self.last_phase = curr_phase
                
            if curr_ext > self.last_ext_count:
                self.event_log.appendleft(f"[{self.env.time:.1f}s] EXTENSION TRIGGERED by high intent")
                self.last_ext_count = curr_ext

            # Trail Logic
            if self.show_trails:
                for p in self.env.pedestrians:
                    if p.id not in self.trails: self.trails[p.id] = deque(maxlen=15)
                    self.trails[p.id].append(world_to_screen(p.x, p.y))
                for v in self.env.vehicles:
                    if v.id not in self.trails: self.trails[v.id] = deque(maxlen=10)
                    self.trails[v.id].append(world_to_screen(v.x, v.y))

    def draw_scene(self):
        self.screen.fill(BG_COLOR)
        
        # Sidewalks
        pygame.draw.rect(self.screen, SIDEWALK_COLOR, (0, 0, OX - 40, HEIGHT))
        pygame.draw.rect(self.screen, SIDEWALK_COLOR, (OX + 40, 0, WIDTH, HEIGHT))
        
        # NS Road
        pygame.draw.rect(self.screen, ROAD_COLOR, (OX - 40, 0, 80, HEIGHT))
        
        # Dashed Center Line
        for y in range(0, HEIGHT, 40):
            pygame.draw.rect(self.screen, YELLOW, (OX - 2, y, 4, 20))
            
        # Stop Lines
        sy_stop = int(OY + 5.0 * SCALE)
        pygame.draw.rect(self.screen, WHITE, (OX - 40, sy_stop, 80, 4))
        
        # Crosswalk Zebra Stripes
        cw_y_min, cw_y_max = int(OY - 12.0 * SCALE), int(OY - 8.0 * SCALE)
        for x in range(OX - 35, OX + 35, 12):
            pygame.draw.rect(self.screen, WHITE, (x, cw_y_min, 6, cw_y_max - cw_y_min))

    def draw_agents(self):
        # Trails
        if self.show_trails:
            for t_pts in self.trails.values():
                if len(t_pts) > 1:
                    pygame.draw.lines(self.screen, (100, 100, 100), False, t_pts, 2)
                    
        # Vehicles
        for v in self.env.vehicles:
            sx, sy = world_to_screen(v.x, v.y)
            v_rect = pygame.Rect(sx - 12, sy - 25, 24, 50)
            pygame.draw.rect(self.screen, WHITE, v_rect, border_radius=4)
            # Headlights/Brakelights
            b_color = RED if v.vy < 2.0 else YELLOW
            pygame.draw.rect(self.screen, b_color, (sx - 10, sy - 25, 6, 4))
            pygame.draw.rect(self.screen, b_color, (sx + 4, sy - 25, 6, 4))
            
            if self.selected_agent and self.selected_agent[0] == "veh" and self.selected_agent[1].id == v.id:
                pygame.draw.rect(self.screen, GREEN, v_rect, 2, border_radius=4)

        # Pedestrians
        for p in self.env.pedestrians:
            sx, sy = world_to_screen(p.x, p.y)
            color = PED_COLORS.get(p.ped_type, WHITE)
            if p.state == "CROSSING": color = GREEN
            
            pygame.draw.circle(self.screen, color, (sx, sy), 7)
            pygame.draw.circle(self.screen, (0,0,0), (sx, sy), 7, 1) # Outline
            
            if p.intent_prob > 0.1:
                txt = self.font.render(f"{p.intent_prob:.1f}", True, (200, 200, 200))
                self.screen.blit(txt, (sx - 10, sy - 20))
                
            if self.selected_agent and self.selected_agent[0] == "ped" and self.selected_agent[1].id == p.id:
                pygame.draw.circle(self.screen, YELLOW, (sx, sy), 10, 2)

        # Traffic Light
        l_color = GREEN if self.env.controller.current_phase == "EW_GREEN" else RED
        pygame.draw.circle(self.screen, (40,40,40), (OX - 60, OY - 70), 16)
        pygame.draw.circle(self.screen, l_color, (OX - 60, OY - 70), 12)

    def draw_hud(self):
        # Background Panel
        panel = pygame.Surface((320, HEIGHT), pygame.SRCALPHA)
        panel.fill(HUD_BG)
        self.screen.blit(panel, (WIDTH - 320, 0))
        
        x_off = WIDTH - 300
        y_off = 20
        
        def blit_text(text, color=WHITE, bold=False):
            nonlocal y_off
            f = self.title_font if bold else self.font
            surf = f.render(text, True, color)
            self.screen.blit(surf, (x_off, y_off))
            y_off += 25

        blit_text("SIMULATION DASHBOARD", GREEN, bold=True)
        blit_text("-" * 35)
        ctrl_name = self.env.controller.__class__.__name__.replace("Controller", "")
        blit_text(f"Controller: {ctrl_name}")
        blit_text(f"Time      : {self.env.time:.1f} s")
        blit_text(f"Phase     : {self.env.controller.current_phase}")
        blit_text(f"Speed     : {self.sim_speed} FPS")
        
        y_off += 10
        blit_text("METRICS", YELLOW, bold=True)
        blit_text("-" * 35)
        blit_text(f"Near Misses: {self.env.near_misses}")
        blit_text(f"Extensions : {getattr(self.env.controller, 'extensions_count', 0)}")
        blit_text(f"Active Peds: {len(self.env.pedestrians)}")
        blit_text(f"Active Vehs: {len(self.env.vehicles)}")

        y_off += 10
        blit_text("EVENT LOG", GREEN, bold=True)
        blit_text("-" * 35)
        for log in self.event_log:
            blit_text(log, (180, 180, 180))
            
        y_off += 20
        blit_text("INSPECTOR", YELLOW, bold=True)
        blit_text("-" * 35)
        if self.selected_agent:
            atype, obj = self.selected_agent
            if atype == "ped":
                blit_text(f"Type  : {obj.ped_type.upper()}")
                blit_text(f"State : {obj.state}")
                blit_text(f"Wait  : {obj.wait_time:.1f} s")
                blit_text(f"Intent: {obj.intent_prob:.2f}")
            elif atype == "veh":
                blit_text(f"Speed : {obj.vy:.1f} m/s")
                blit_text(f"Delay : {obj.delay_time:.1f} s")
        else:
            blit_text("Click an agent to inspect", (150, 150, 150))
            
        y_off = HEIGHT - 100
        blit_text("[SPACE] Play/Pause", (150,150,150))
        blit_text("[UP/DN] Speed", (150,150,150))
        blit_text("[ C ] Cycle Control", (150,150,150))

    def run(self):
        running = True
        while running:
            running = self.handle_events()
            self.update_logic()
            
            self.draw_scene()
            self.draw_agents()
            self.draw_hud()
            
            pygame.display.flip()
            self.clock.tick(self.sim_speed)
            
        pygame.quit()

if __name__ == "__main__":
    vis = Visualizer()
    vis.run()