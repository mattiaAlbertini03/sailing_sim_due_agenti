# ==========================================
# ⚙ CONFIGURAZIONI DEL SIMULATORE 
# ==========================================

# --- FISICA E CAMPO DI GARA ---
FIELD_SIZE = 500             # Dimensione del campo (500x500)
MAX_STEPS = 1000             # Numero massimo di step prima del timeout
DT = 1.0                     # Delta time (scorrimento del tempo)
TARGET_RADIUS = 50.0         # Raggio utile per "toccare" la boa
FINISH_LINE_Y = 150.0         # Coordinata Y della linea di traguardo

# --- BARCA E VENTO ---
MAX_BOAT_SPEED = 15.0        # Velocità massima della barca
MAX_WIND_SPEED = 25.0        # Velocità massima assoluta del vento
WIND_MIN_RANDOM = 12.0       # Vento minimo generato a inizio gara
WIND_MAX_RANDOM = 18.0       # Vento massimo generato a inizio gara

# --- HYPERPARAMETERS DELL'AGENTE (PPO) ---
TOTAL_TIMESTEPS_SOLO = 3000000    # Durata totale dell'addestramento in solo
TOTAL_TIMESTEPS_SELF_PLAY = 500000 # Durata totale dell'addestramento in self-play
LEARNING_RATE = 3e-4         # Tasso di apprendimento
N_STEPS = 2048               # Step raccolti prima di aggiornare la rete
BATCH_SIZE = 64              # Dimensione del batch per la rete neurale
N_EPOCHS = 10                # Epoche di ottimizzazione
GAMMA = 0.99                 # Fattore di sconto (lungimiranza dell'agente)
GAE_LAMBDA = 0.95            # Smoothing dei vantaggi
CLIP_RANGE = 0.2             # Limite di variazione della policy
ENT_COEF = 0.005             # Entropia (più è alta, più l'agente "esplora" a caso)

# --- SALVATAGGI, LOG E VIDEO ---
CHECK_FREQ = 10000           # Ogni quanti step stampare i progressi a schermo
MODEL_NAME = "models/ppo_match_race"
TENSORBOARD_LOG_DIR = "logs/sailing_tensorboard/"
VIDEO_NAME = "match_race.mp4"
VIDEO_FPS = 15
MAX_VIDEO_ATTEMPTS = 10      # Quante volte provare a generare un video vincente
