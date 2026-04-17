import numpy as np

import config

def ccw(A, B, C):
    """Funzione di supporto per l'intersezione tra due segmenti."""
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

def check_intersection(pos_prec, pos, boa1, boa2):
    """Controlla se il segmento pos_prec->pos interseca il segmento della linea di traguardo/cancello."""
    return ccw(pos_prec, boa1, boa2) != ccw(pos, boa1, boa2) and ccw(pos_prec, pos, boa1) != ccw(pos_prec, pos, boa2)

def normalize_angle(angle):
    """Mantiene l'angolo compreso tra 0 e 2*pi."""
    return angle % (2 * np.pi)

def get_polar_speed(wind_angle, wind_speed):
    """Calcola la velocità target della barca in base all'angolo del vento e all'efficienza aerodinamica (Polari)."""
    wind_angle = normalize_angle(wind_angle + np.pi)
    angle_deg = np.abs(np.degrees(wind_angle))
    if angle_deg > 180: angle_deg = 360 - angle_deg

    if angle_deg < 40: speed_ratio = 0.0
    elif angle_deg < 50: speed_ratio = 0.2 + (angle_deg - 40) * 0.02
    elif angle_deg < 90: speed_ratio = 0.4 + (angle_deg - 50) * 0.0075
    elif angle_deg < 120: speed_ratio = 0.7
    elif angle_deg < 150: speed_ratio = 0.7 - (angle_deg - 120) * 0.003
    else: speed_ratio = 0.6 - (angle_deg - 150) * 0.005
    
    return min(speed_ratio * wind_speed, config.MAX_BOAT_SPEED)
