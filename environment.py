import numpy as np
import gymnasium as gym
from gymnasium import spaces
from pettingzoo import ParallelEnv
import config

from render_utils import render_frame 

# Importiamo i nostri moduli di fisica e regole
from physics import WindField, SailingBoat, calculate_wind_shadow
from rules import check_penalties
from utils import check_intersection, normalize_angle

class ImprovedSailingEnv(ParallelEnv):
    """
    Ambiente Multi-Agente per il Match Race della Coppa America.
    Supporta 2 barche che competono applicando regole di precedenza e wind shadow.
    """
    def __init__(self, render_mode=None, mode="solo"):
        super().__init__()
        self.render_mode = render_mode
        self.mode = mode
        
        # Definiamo gli agenti in base alla modalità
        if self.mode == "self_play":
            self.agents = ["boat_0", "boat_1"]
        else:
            self.agents = ["boat_0"]
        
        self.field_size_x = config.FIELD_SIZE
        self.field_size_y = config.FIELD_SIZE + 100
        self.target_radius = 50.0
        self.max_steps = config.MAX_STEPS
        self.finish_line_y = config.FINISH_LINE_Y
        
        # Dizionari per spazi di azione e osservazione
        self.action_spaces = {}
        self.observation_spaces = {}
        
        for agent in self.agents:
            # Action: [Timone (-1 a 1), Foil (-1 a 1)]
            self.action_spaces[agent] = spaces.Box( low=np.array([-1.0, -1.0]), high=np.array([1.0, 1.0]),  dtype=np.float32 )
            
            # Obs: 13 valori (5 propri + 2 vento + 2 target + 4 nemico)
            self.observation_spaces[agent] = spaces.Box( low=-np.inf, high=np.inf, shape=(13,), dtype=np.float32 )
            
        self.wind = None
        self.boats = {}
        self.gates = []
        self.step_count = 0


    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)

        self.step_count = 0
        
        self.wind = WindField()
        _, self.start_wind_dir = self.wind.get_wind_at(self.field_size_x/2, self.field_size_y/2)
        
        # Posizioniamo SEMPRE la barca 0
        self.boats = {"boat_0": SailingBoat(boat_id="boat_0")}
        self.boats["boat_0"].heading = np.pi # Punta verso Ovest
        self.boats["boat_0"].speed = 5.0
        
        # Posizioniamo la barca 1 SOLO in self-play
        if self.mode == "self_play":
            self.boats["boat_1"] = SailingBoat(boat_id="boat_1")
            self.boats["boat_1"].heading = 0.0       # Punta verso Est 
            self.boats["boat_1"].speed = 5.0
        
        # Gate a bastone
        target_1 = np.array([np.random.uniform(self.field_size_x - 300, self.field_size_x - 200),
                             np.random.uniform(self.field_size_y - 100, self.field_size_y - 50)])
        target_2 = np.array([np.random.uniform(self.field_size_x - 300, self.field_size_x - 200),
                             np.random.uniform(self.field_size_y - 400, self.field_size_y - 350)])
        self.gates = np.array([target_1, target_2])

        # Tracker per i bonus del primo arrivato: un flag per ogni cancello + 1 per il traguardo finale
        self.gate_claimed = [False] * (len(self.gates) + 1)

        observations = {agent: self._get_obs(agent) for agent in self.agents}
        infos = {agent: {} for agent in self.agents}
        
        return observations, infos

    def step(self, actions):
        self.step_count += 1
        
        rewards = {agent: 0.0 for agent in self.agents}
        terminations = {agent: False for agent in self.agents}
        truncations = {agent: False for agent in self.agents}
        dist_to_target = {agent: 999 for agent in self.agents}
        
        # 1. EVOLUZIONE VENTO
        self.wind.step()
        _, global_wind_dir = self.wind.get_wind_at(self.field_size_x/2, self.field_size_y/2)

        # 2. CALCOLO WIND SHADOW
        b0 = self.boats["boat_0"]
        w_speed_0_base, w_dir_0 = self.wind.get_wind_at(b0.x, b0.y)
        b0.update_local_wind(w_speed_0_base, w_dir_0)
        
        if self.mode == "self_play":
            b1 = self.boats["boat_1"]
            w_speed_1_base, w_dir_1 = self.wind.get_wind_at(b1.x, b1.y)
            b1.update_local_wind(w_speed_1_base, w_dir_1)
            
            # Applichiamo i rifiuti prima di muovere le barche
            w_speed_0, w_speed_1 = calculate_wind_shadow(b0, b1, global_wind_dir)
            
            # --- BONUS COPERTURA (WIND SHADOWING) ---
            # Se la barca 1 ha perso vento rispetto al suo base, la barca 0 la sta coprendo!
            if w_speed_1 < w_speed_1_base - 0.5: # Ha perso almeno mezzo nodo
                if "boat_0" in rewards: rewards["boat_0"] += 0.5
                
            # E viceversa: se la barca 0 ha perso vento, la barca 1 la sta coprendo
            if w_speed_0 < w_speed_0_base - 0.5:
                if "boat_1" in rewards: rewards["boat_1"] += 0.5
        else:
            w_speed_0 = w_speed_0_base


        # 3. AGGIORNAMENTO FISICA SINGOLE BARCHE
        for agent in self.agents:
            if terminations[agent]:
                continue
            boat = self.boats[agent]
            action = actions[agent]
            
            # Assegna il vento corretto (se è solo, agent sarà sempre boat_0)
            wind_spd = w_speed_0 if agent == "boat_0" else w_speed_1
            
            target = None
            if boat.gate_index >= len(self.gates):
                target = np.array([boat.x, config.FINISH_LINE_Y])
            else:
                target = self.gates[boat.gate_index]
            
            start = False
            
            pos_prec = np.array([boat.x, boat.y])
            prev_dist = np.linalg.norm(target - pos_prec)
        
            if boat.y > self.finish_line_y and boat.gate_index < len(self.gates):
                start = True

            rewards[agent] += boat.update_physics(action[0], action[1], wind_spd)

            pos= np.array([boat.x, boat.y])

            current_dist = np.linalg.norm(target - pos)
            dist_to_target[agent] = current_dist

            rewards[agent] -= abs(action[0]) * 0.01
            if boat.speed < 1.0:
                rewards[agent] -= 2


            angle_to_target = np.arctan2(target[1]-boat.y, target[0]- boat.x)
            
            # VMG verso la boa
            heading_error = abs(normalize_angle(angle_to_target - boat.heading))
            vmg = boat.speed * np.cos(heading_error)
            rewards[agent] += vmg * 0.01 

            if boat.speed > 4.0:
                rewards[agent] += 0.05
            
           # --- PENALITÀ NO-GO ZONE (Controvento) ---
            wind_diff = abs((boat.heading - boat.wind_dir + np.pi) % (2 * np.pi) - np.pi)
            if abs(wind_diff - np.pi) < np.radians(45):
                rewards[agent] -= 2  # Punizione netta: ti stai fermando!

            angle_deg = np.degrees(wind_diff)
            is_downwind = angle_deg < 80
            if boat.foil:
                rewards[agent] += 0.5
            elif is_downwind and boat.speed >= 4.0:
                #se il vento è a favore e non stai volando -> penalità
                rewards[agent] -= 0.5

            # --- BONUS VELOCITÀ PURO ---
            # Incoraggia a mantenere il flusso laminare sulle vele
            rewards[agent] += (boat.speed / config.MAX_BOAT_SPEED) * 0.15

            target_reached = False

            if boat.gate_index == len(self.gates):
                if boat.y <= config.FINISH_LINE_Y: target_reached = True
            else:
                perp_angle = self.start_wind_dir + (np.pi / 2)
                dx = (self.target_radius-5) * np.cos(perp_angle)
                dy = (self.target_radius-5) * np.sin(perp_angle)
            
                puntoA = (target[0] - dx, target[1] - dy)
                puntoB = (target[0] + dx, target[1] + dy)

                diff_dir = np.cos(normalize_angle(boat.wind_dir-boat.heading))
                intersected = check_intersection(pos_prec, pos, puntoA, puntoB)

                if intersected and (((not boat.gate_index) and diff_dir < 0) or (boat.gate_index and diff_dir > 0)):
                    target_reached = True

            if target_reached:
                efficiency = max(0, self.max_steps - self.step_count)
                current_gate = boat.gate_index
                boat.gate_index += 1
                rewards[agent] += (1000.0 * boat.gate_index) + efficiency



                if self.mode == "self_play" and not self.gate_claimed[current_gate]:
                    # È il primo ad arrivare a questo cancello!
                    self.gate_claimed[current_gate] = True
                    rewards[agent] += 2000.0  # Mega bonus per aver battuto l'avversario
                    
                    # Se questo era l'arrivo finale, dichiariamo la vittoria
                    if boat.gate_index > len(self.gates):
                        for a in self.agents:
                            terminations[a] = True # Finiamo la partita per tutti
                            if a != agent:
                                rewards[a] -= 1000.0 # Penalità di sconfitta al perdente
                
                # Comportamento standard per la modalità Solo (chiusura regata a fine boe)
                elif boat.gate_index > len(self.gates):
                    terminations[agent] = True

            # Penalità Uscita dal Campo
            lim_y_inf = self.finish_line_y if start else 0
            if boat.x < 0 or boat.x > self.field_size_x or boat.y > self.field_size_y or boat.y < lim_y_inf:
                rewards[agent] -= 500.0        # Mazzata letale!
                terminations[agent] = True     # L'episodio finisce qui, hai perso.

        # Condizione di fine tempo (Truncation)
        if self.step_count >= self.max_steps:
            for agent in self.agents:
                truncations[agent] = True
        
        # 4. CONTROLLO REGOLE E COLLISIONI (Arbitro)
        if self.mode == "self_play":
            pen_0, pen_1 = check_penalties(b0, b1, global_wind_dir)
            if "boat_0" in rewards:
                rewards["boat_0"] += pen_0
            if "boat_1" in rewards:
                rewards["boat_1"] += pen_1

        observations = {agent: self._get_obs(agent) for agent in self.agents}

        infos = { agent: {
            'distance_to_target': dist_to_target[agent],
            'indextarget': self.boats[agent].gate_index,
            'speed': self.boats[agent].speed,
            'steps': self.step_count
            } for agent in self.agents }

        return observations, rewards, terminations, truncations, infos

    def _get_obs(self, agent_id):
        """Costruisce l'osservazione includendo i dati del nemico (reale o fantasma)."""
        boat = self.boats[agent_id]
        
        # FIX SICUREZZA: Evitiamo crash se la barca ha superato l'ultimo gate
        if boat.gate_index < len(self.gates):
            target = self.gates[boat.gate_index]
        else:
            target = np.array([boat.x, config.FINISH_LINE_Y])
            
        pos = np.array([boat.x, boat.y])
        diff = target - pos
        dist_to_target = np.linalg.norm(diff)
        angle_to_target = np.arctan2(diff[1], diff[0])
        
        local_wind_spd, local_wind_dir = boat.wind_speed, boat.wind_dir
        dist_max = np.sqrt(self.field_size_x**2 + self.field_size_y**2)
        
        # --- GENERAZIONE NEMICO (Reale o Fantasma) ---
        if self.mode == "self_play":
            enemy_id = "boat_1" if agent_id == "boat_0" else "boat_0"
            enemy = self.boats[enemy_id]
            rel_enemy_x = enemy.x - boat.x
            rel_enemy_y = enemy.y - boat.y
            rel_enemy_angle = (enemy.heading - boat.heading + np.pi) % (2 * np.pi) - np.pi
            enemy_speed = enemy.speed
        else:
            rel_enemy_x = self.field_size_x
            rel_enemy_y = self.field_size_y
            rel_enemy_angle = 0.0
            enemy_speed = 0.0
        
        # --- ANGOLI RELATIVI ---
        rel_wind_angle = (local_wind_dir - boat.heading + np.pi) % (2 * np.pi) - np.pi
        rel_target_angle = (angle_to_target - boat.heading + np.pi) % (2 * np.pi) - np.pi
        
        obs = np.array([
            boat.x / self.field_size_x,
            boat.y / self.field_size_y,
            boat.heading,                         
            boat.speed / config.MAX_BOAT_SPEED,
            1.0 if boat.foil else 0.0,
            local_wind_spd / config.MAX_WIND_SPEED,
            rel_wind_angle,                       # Vento: 0 = in faccia, pi/2 = da sinistra
            dist_to_target / dist_max,
            rel_target_angle,                     # Boa: 0 = dritta davanti
            rel_enemy_x / self.field_size_x,
            rel_enemy_y / self.field_size_y,
            rel_enemy_angle,                      # Orientamento nemico
            enemy_speed / config.MAX_BOAT_SPEED   # Uso la variabile enemy_speed
        ], dtype=np.float32)
        
        return obs
    
    def render(self):
        if self.render_mode == 'rgb_array': 
            return render_frame(self)

    def close(self):
        pass
    
    def observation_space(self, agent):
        return self.observation_spaces[agent]