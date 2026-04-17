import numpy as np
from utils import normalize_angle


def get_tack(boat_heading, wind_dir):
    """
    Determina le mure della barca (da che lato riceve il vento).
    wind_dir: direzione DA CUI soffia il vento.
    Restituisce: 'starboard' (mure a dritta) o 'port' (mure a sinistra).
    """
    # Calcoliamo l'angolo relativo del vento rispetto alla prua della barca
    # Normalizziamo l'angolo tra -pi e pi
    relative_wind = normalize_angle(wind_dir - boat_heading)
    
    # Se il vento arriva da destra (angolo positivo), siamo mure a dritta (starboard)
    # Se arriva da sinistra (angolo negativo), siamo mure a sinistra (port)
    if relative_wind >= 0:
        return 'starboard'
    else:
        return 'port'

def is_windward(boat_a, boat_b, wind_dir):
    """
    Determina se boat_a è sopravento (windward) rispetto a boat_b.
    Usa il prodotto scalare per proiettare la posizione sul vettore del vento.
    """
    # Vettore che va dalla barca B alla barca A
    dx = boat_a.x - boat_b.x
    dy = boat_a.y - boat_b.y
    
    # Vettore della direzione del vento (da dove arriva)
    wind_vec_x = np.cos(wind_dir)
    wind_vec_y = np.sin(wind_dir)
    
    # Prodotto scalare: se è positivo, A è più "vicino" alla sorgente del vento rispetto a B
    dot_product = dx * wind_vec_x + dy * wind_vec_y
    
    return dot_product > 0

def get_right_of_way(boat_1, boat_2, wind_dir):
    """
    Arbitro virtuale: calcola chi ha la precedenza tra due barche.
    Restituisce l'ID della barca che ha il diritto di rotta (Right of Way).
    """
    tack_1 = get_tack(boat_1.heading, wind_dir)
    tack_2 = get_tack(boat_2.heading, wind_dir)
    
    # REGOLA 10 DELLA VELA: Mure diverse. Mure a dritta (starboard) ha la precedenza.
    if tack_1 == 'starboard' and tack_2 == 'port':
        return boat_1.id
    elif tack_2 == 'starboard' and tack_1 == 'port':
        return boat_2.id
        
    # REGOLA 11 DELLA VELA: Stesse mure. La barca sottovento (leeward) ha la precedenza.
    if tack_1 == tack_2:
        # Se la barca 1 è sopravento, la barca 2 ha la precedenza (e viceversa)
        if is_windward(boat_1, boat_2, wind_dir):
            return boat_2.id
        else:
            return boat_1.id
            
    return None # Fallback di sicurezza

def check_penalties(boat_1, boat_2, wind_dir, penalty_radius=30.0):
    """
    Controlla se c'è un'infrazione e distribuisce le penalità in modo proporzionale.
    Restituisce una tupla: (penalità_barca_1, penalità_barca_2)
    """
    dist = np.hypot(boat_1.x - boat_2.x, boat_1.y - boat_2.y)
    
    # Se le barche sono a distanza di sicurezza, nessuna penalità
    if dist > penalty_radius:
        return 0.0, 0.0
        
    # Scopriamo chi ha ragione e chi ha torto
    right_of_way_id = get_right_of_way(boat_1, boat_2, wind_dir)
    
    # Calcoliamo la gravità: più sono vicine sotto i 30 metri, più la penalità sale
    # Questo aiuta il "Reward Shaping" dell'agente a capire il gradiente di errore
    severity = (penalty_radius - dist) / penalty_radius
    base_penalty = 100.0 * severity  # Penalità massima di 100 punti se si scontrano
    
    penalty_1 = 0.0
    penalty_2 = 0.0
    
    # Chi non ha la precedenza si prende la penalità negativa
    if right_of_way_id == boat_1.id:
        penalty_2 = -base_penalty
    elif right_of_way_id == boat_2.id:
        penalty_1 = -base_penalty
        
    return penalty_1, penalty_2
