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
    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode
        
        # Definiamo gli agenti
        self.agents= ["boat_0", "boat_1"]
        
        self.field_size_x = config.FIELD_SIZE
        self.field_size_y = config.FIELD_SIZE + 100
        self.target_radius = 50.0
        self.max_steps = config.MAX_STEPS
        self.finish_line_y = config.FINISH_LINE_Y
        
        # Dizionari per spazi di azione e osservazione
        self.action_spaces = {}
        self.observation_spaces = {}
        
        for agent in self.agents:
            # Action: [Timone (-1 a 1), Foil (0 a 1)]
            self.action_spaces[agent] = spaces.Box( low=np.array([-1.0, 0.0], dtype=np.float32), high=np.array([1.0, 1.0], dtype=np.float32),  dtype=np.float32 )
            
            # Obs: 13 valori (5 propri + 2 vento + 2 target + 4 nemico)
            self.observation_spaces[agent] = spaces.Box( low=-np.inf, high=np.inf, shape=(13,), dtype=np.float32 )
            
        self.wind = None
        self.boats = {}
        self.gates = []
        self.step_count = 0


    def reset(self, seed=None, options=None):
        #rimossa questa riga per evitare errori di reset in caso di selfplay
        #super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)

        self.step_count = 0
        
        self.wind = WindField()
        _, self.start_wind_dir = self.wind.get_wind_at(self.field_size_x/2, self.field_size_y/2)
        
        # Posizioniamo le barche separate sulla linea di partenza
        self.boats = {
            "boat_0": SailingBoat(boat_id="boat_0"),
            "boat_1": SailingBoat(boat_id="boat_1")
        }

        #orientamento iniziale per favorire la presa di velocità
        self.boats["boat_0"].heading = (np.pi / 2) - (np.pi / 4)
        self.boats["boat_1"].heading = (np.pi / 2) + (np.pi / 4)
        
        # Gate a bastone
        target_1 = np.array([np.random.uniform(self.field_size_x - 300, self.field_size_x - 200),
                             np.random.uniform(self.field_size_y - 100, self.field_size_y - 50)])
        target_2 = np.array([np.random.uniform(self.field_size_x - 300, self.field_size_x - 200),
                             np.random.uniform(self.field_size_y - 400, self.field_size_y - 350)])
        self.gates = np.array([target_1, target_2])

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
        
        # Vento al centro del campo (utile per le regole generali)
        _, global_wind_dir = self.wind.get_wind_at(self.field_size_x/2, self.field_size_y/2)

        # 2. CALCOLO WIND SHADOW (Fisica condivisa)
        b0, b1 = self.boats["boat_0"], self.boats["boat_1"]
        w_speed_0, w_dir_0 = self.wind.get_wind_at(b0.x, b0.y)
        w_speed_1, w_dir_1 = self.wind.get_wind_at(b1.x, b1.y)
        
        b0.update_local_wind(w_speed_0, w_dir_0)
        b1.update_local_wind(w_speed_1, w_dir_1)

        # Applichiamo i rifiuti prima di muovere le barche
        w_speed_0, w_speed_1 = calculate_wind_shadow(b0, b1, global_wind_dir)

        # 3. AGGIORNAMENTO FISICA SINGOLE BARCHE
        for agent in self.agents:
            if terminations[agent]:
                continue
            boat = self.boats[agent]
            action = actions[agent]
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


           # Usiamo current_dist (la variabile locale definita poche righe sopra) -> penalità ridotta se la barca è ferma lontano dal target
            """if boat.speed < 0.5:
                rewards[agent] -= 0.1 * (current_dist / 500.0)
            # 1. Tassa di base per avere il timone piegato (così preferisce andare dritto)
            rewards[agent] -= abs(action[0]) * 0.01

            # Calcoliamo lo sbalzo
            rudder_change = abs(action[0] - boat.prev_action)

            # 2. La tua struttura a cascata:
            if rudder_change > 1.5:
                rewards[agent] -= 1.0  # Mazzata letale per il tremolio estremo (es. +0.8 a -0.9)
            elif rudder_change > 0.5:
                # 3. Tassa per sbalzi medi/bruschi
                rewards[agent] -= rudder_change * 0.05
            elif rudder_change > 0.05:
                # 4. ZONA NEUTRA: La barca sta curvando in modo dolce (es. da 0.0 a 0.2). 
                # Non le diamo multe, ma non le regaliamo nemmeno il bonus stabilità.
                pass 
            else:
                # 5. Bonus per mantenere il timone DAVVERO stabile (micro-correzioni o fermo)
                rewards[agent] += 0.01""
                
            boat.prev_action = action[0]"""

            angle_to_target = np.arctan2(target[1]-boat.y, target[0]- boat.x)
            
            # VMG verso la boa
            heading_error = abs(normalize_angle(angle_to_target - boat.heading))
            vmg = boat.speed * np.cos(heading_error)
            rewards[agent] += vmg * 0.01 
           
           # --- PENALITÀ NO-GO ZONE (Controvento) ---
            # Calcoliamo la differenza angolare minima tra la prua e il vento
            wind_diff = abs((boat.heading - boat.wind_dir + np.pi) % (2 * np.pi) - np.pi)
            
            # Se la differenza è vicina a 180 gradi (pi), abbiamo il vento in faccia.
            # np.radians(45) crea un "cono" di 45 gradi per lato in cui la barca prende la penalità.
            if abs(wind_diff - np.pi) < np.radians(45):
                rewards[agent] -= 0.1  # Fastidio continuo per chi naviga controvento

            
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

                """if boat.gate_index:
                    rewards[agent] += diff_dir * 0.1
                else:
                    rewards[agent] += diff_dir * -0.1"""
                intersected = check_intersection(pos_prec, pos, puntoA, puntoB)

                #per ora semplifichiamo
                if intersected and (((not boat.gate_index) and diff_dir < 0) or (boat.gate_index and diff_dir > 0)):
                    target_reached = True

            if target_reached:
                efficiency = max(0, self.max_steps - self.step_count)
                boat.gate_index += 1
                rewards[agent] += (1000.0 * boat.gate_index) + efficiency
                if boat.gate_index > len(self.gates):
                    terminations[agent] = True

            # Penalità Uscita dal Campo
            lim_y_inf = self.finish_line_y if start else 0
            if boat.x < 0 or boat.x > self.field_size_x or boat.y > self.field_size_y or boat.y < lim_y_inf:
                # 1. Abbassiamo la penalità da -500 a un valore fastidioso ma non letale
                rewards[agent] -= 5.0
                
                # 2. Rimuoviamo terminated = True (l'episodio NON finisce)
                
                # 3. Forziamo la barca a rimanere fisicamente dentro i limiti
                boat.x = np.clip(boat.x, 0, self.field_size_x)
                boat.y = np.clip(boat.y, lim_y_inf, self.field_size_y)
            # -------------------------------------------------------------
#TODO
# CODICE DI CAROL
# da mettere per non avere errori nelle penalità (forse, da vedere)
                #boat.x = np.clip(boat.x, 0, self.field_size)
                #boat.y = np.clip(boat.y, 0, self.field_size)


        # Condizione di fine tempo (Truncation)
        if self.step_count >= self.max_steps:
            for agent in self.agents:
                truncations[agent] = True
        
        # 4. CONTROLLO REGOLE E COLLISIONI (Arbitro)
        pen_0, pen_1 = check_penalties(b0, b1, global_wind_dir)

        #applica la penalità solo se la barca è ancora nel dizionnario dei rewards
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
        """Costruisce l'osservazione includendo i dati del nemico."""
        boat = self.boats[agent_id]
        enemy_id = "boat_1" if agent_id == "boat_0" else "boat_0"
        enemy = self.boats[enemy_id]
        
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
        
        # Calcoliamo posizione relativa del nemico
        rel_enemy_x = enemy.x - boat.x
        rel_enemy_y = enemy.y - boat.y
        dist_max = np.sqrt(self.field_size_x**2 + self.field_size_y**2)
        
        # --- LA MAGIA: ANGOLI RELATIVI ---
        # Sottraiamo l'orientamento della barca agli altri angoli
        rel_wind_angle = normalize_angle(local_wind_dir - boat.heading)
        rel_target_angle = normalize_angle(angle_to_target - boat.heading)
        rel_enemy_angle = normalize_angle(enemy.heading - boat.heading)
        
        obs = np.array([
            boat.x / self.field_size_x,
            boat.y / self.field_size_y,
            boat.heading,                         # 3. Manteniamo l'assoluto per fargli capire i bordi mappa
            boat.speed / config.MAX_BOAT_SPEED,
            1.0 if boat.foil else 0.0,
            local_wind_spd / config.MAX_WIND_SPEED,
            rel_wind_angle,                       # 7. Vento: 0 = in faccia, pi/2 = da sinistra
            dist_to_target / dist_max,
            rel_target_angle,                     # 9. Boa: 0 = dritta davanti a noi
            rel_enemy_x / self.field_size_x,
            rel_enemy_y / self.field_size_y,
            rel_enemy_angle,                      # 12. Orientamento nemico: 0 = parallelo a noi
            enemy.speed / config.MAX_BOAT_SPEED
        ], dtype=np.float32)
        
        return obs
    
    def render(self):
        if self.render_mode == 'rgb_array': 
            return render_frame(self)

    def close(self):
        pass # La logica di chiusura della figura la gestiamo in render_utils se serve
    
    def observation_space(self, agent):
        return self.observation_spaces[agent]
