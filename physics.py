import numpy as np
import math

from utils import get_polar_speed
from utils import normalize_angle
import config

class WindField:
    """
    Gestisce un campo di vento 2D con random walk spaziale e temporale.
    Invece di avere un vento globale, divide il campo in una griglia.
    """
    def __init__(self):
        self.field_size = config.FIELD_SIZE 
        self.grid_res = 100
        self.grid_size = int(self.field_size / self.grid_res) + 1
        
        base_dir = (3 * np.pi / 2 ) + np.random.uniform(-np.pi/12, np.pi/12)
        #base_dir = (np.pi / 2) + np.random.uniform(-np.pi/12, np.pi/12) # Vento da poppa (a favore)
        base_speed = np.random.uniform(config.WIND_MIN_RANDOM, config.WIND_MAX_RANDOM)
        
        # Inizializziamo la griglia del vento (velocità e direzione per ogni cella)

        #TODO
        #dovremmo creare grid_size_x e grid_size_y per gestire campi non quadrati
        self.wind_speed_grid = np.full((self.grid_size, self.grid_size), base_speed)
        self.wind_dir_grid = np.full((self.grid_size, self.grid_size), base_dir)
        
    def step(self):
        """Evolve il vento nel tempo aggiungendo rumore stocastico (raffiche)."""
        speed_noise = np.random.uniform(-0.2, 0.2, (self.grid_size, self.grid_size))
        dir_noise = np.random.uniform(-0.05, 0.05, (self.grid_size, self.grid_size))
        
        self.wind_speed_grid = np.clip(self.wind_speed_grid + speed_noise, 8.0, config.MAX_WIND_SPEED)
        
        # Normalizziamo l'angolo
        self.wind_dir_grid = normalize_angle(self.wind_dir_grid + dir_noise)

    def get_wind_at(self, x, y):
        """Restituisce il vento locale interpolato per una coordinata x, y."""
        # Troviamo gli indici della griglia più vicini
        grid_x = int(np.clip(x / self.grid_res, 0, self.grid_size - 1))
        grid_y = int(np.clip(y / self.grid_res, 0, self.grid_size - 1))
        
        return self.wind_speed_grid[grid_x, grid_y], self.wind_dir_grid[grid_x, grid_y]



class SailingBoat:
    """
    Gestisce la fisica, la posizione e l'attrito (drag) della barca.
    """
    def __init__(self, boat_id):
        self.id = boat_id #per distinguere le barche
        self.x = np.random.uniform(10, 490)
        self.y = np.random.uniform(30, 120)
        self.wind_speed =0.0
        self.wind_dir = 0.0
        self.gate_index = 0
        self.heading = np.random.uniform(0, 2*np.pi)
        self.speed = 0.0
        self.max_speed = config.MAX_BOAT_SPEED
        self.foil = False
        self.trajectory = [(self.x, self.y, self.foil)]
        self.prev_action = 0.0
        self.last_foil_action = False

    def get_polar_speed_2(self, apparent_wind_angle, wind_speed):
        """
        Funzione polare continua (Onda sinusoidale).
        Niente più gradini e 'muri invisibili' controvento.
        """
        # La barca va più veloce al traverso/lasco
        efficiency = np.sin(abs(apparent_wind_angle) / 2.0) 
        target = wind_speed * efficiency * 1.5
        return np.clip(target, 0, self.max_speed)

    def update_local_wind(self, local_wind_speed, local_wind_dir):
        self.wind_speed = local_wind_speed
        self.wind_dir = local_wind_dir

    def update_physics(self, action_turn, action_foil, local_wind_speed):
        """
        Aggiorna la fisica in base alle azioni (continue) e al vento locale.
        - action_turn: float tra -1.0 (sx) e 1.0 (dx)
        - action_foil: boolean o float > 0.5 per indicare volontà di volo
        """
        # 1. GESTIONE TIMONE
        rudder_effectiveness = np.clip(self.speed / 3.0, 0.2, 1.0)
        max_turn_rate = np.radians(4) * config.DT
        
        turn_angle = action_turn * max_turn_rate * rudder_effectiveness
        self.heading = normalize_angle(self.heading + turn_angle)
        reward = 0


        
        # 2. GESTIONE FOIL CON ANTI-SINGHIOZZO
        if self.last_foil_action:
            wants_to_foil = action_foil > -0.2
        else:
            wants_to_foil = action_foil > -0.2
        
        self.last_foil_action = wants_to_foil

        if wants_to_foil and self.speed >= 4.0:
            self.foil = True
        else:
            self.foil = False
        
        # 3. CALCOLO VENTO APPARENTE
        apparent_wind_angle = normalize_angle((self.wind_dir - self.heading))

        # 4. CALCOLO VELOCITÀ E INERZIA
        target_speed = get_polar_speed(apparent_wind_angle, local_wind_speed)
        
        # Moltiplicatori del Foil
        if self.foil:
            target_speed *= 1.5  
            turn_penalty = 0.02   
        else:
            target_speed *= 0.6  
            turn_penalty = 0.005   
            
        # Applichiamo la frenata causata dalla virata
        drag_penalty = turn_penalty * abs(action_turn)
        self.speed *= (1.0 - drag_penalty * config.DT)

        if target_speed > self.speed:
            inertia = 0.2   # Accelera reattivamente sotto raffica
        else:
            inertia = 0.04  # Frena MOOOLTO lentamente. La barca "scivola" e conserva il momento per superare l'angolo morto in curva!
            
        self.speed = self.speed + inertia * (target_speed - self.speed)
        self.speed = max(0.0, self.speed)
        
        # 5. AGGIORNAMENTO POSIZIONE X, Y
        displacement = self.speed * config.DT
        self.x += displacement * np.cos(self.heading)
        self.y += displacement * np.sin(self.heading)
        

#TODO
#cosa fatta da me per ridurre il peso nel training
        #if self.render_mode == 'rgb_array':
        self.trajectory.append((self.x, self.y, self.foil))
    
        #Time Penalty (Costante)
        reward -= 0.05

        return reward


def calculate_wind_shadow(boat1, boat2,  wind_dir):
    """
     Calcola l'interferenza del vento tra due barche.
    Restituisce le nuove velocità del vento tenendo conto del cono d'ombra.
    """
    dx = boat2.x - boat1.x
    dy = boat2.y - boat1.y
    dist = np.hypot(dx, dy)
        
    # Se sono a più di 100 metri, nessuna interferenza
    if dist > 100.0:
        return boat1.wind_speed, boat2.wind_speed
            
    angle_1_to_2 = np.arctan2(dy, dx)
    angle_2_to_1 = np.arctan2(-dy, -dx)
        
    # Calcoliamo verso dove SOFFIA il vento (wind_dir è da dove arriva)
    wind_down_angle = normalize_angle(wind_dir)
           
    # Ampiezza del cono d'ombra (es. 20 gradi a destra e sinistra)
    cone_angle = np.radians(20) 
        
    # Calcoliamo se una barca è allineata con il vento rispetto all'altra
    diff_angle_1 = abs(normalize_angle(angle_1_to_2 - wind_down_angle))
    diff_angle_2 = abs(normalize_angle(angle_2_to_1 - wind_down_angle))
        
    # Più le barche sono vicine, più i "rifiuti" sono forti (max 40% di perdita)
    shadow_factor = max(0.0, 1.0 - (dist / 100.0)) * 0.4 
        
    # Inizializza i valori base prima dei controlli
    wind_speed_1 = boat1.wind_speed
    wind_speed_2 = boat2.wind_speed

    if diff_angle_1 < cone_angle:
        # Barca 1 è sopravento: copre la Barca 2
        wind_speed_2 = boat2.wind_speed * (1.0 - shadow_factor)
    elif diff_angle_2 < cone_angle:
        # Barca 2 è sopravento: copre la Barca 1
        wind_speed_1 = boat1.wind_speed * (1.0 - shadow_factor)
            
    return wind_speed_1, wind_speed_2
