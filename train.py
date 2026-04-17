import os
import glob
import random
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
import gymnasium as gym

# Importiamo l'ambiente e le configurazioni
from environment import ImprovedSailingEnv
import config

class SelfPlayWrapper(gym.Env):
    def __init__(self, mode="solo"):
        super().__init__()
        self.mode = mode
        self.env = ImprovedSailingEnv(mode=self.mode)
        
        self.action_space = self.env.action_spaces["boat_0"]
        self.observation_space = self.env.observation_spaces["boat_0"]
        
        self.opponent_model = None
        self.episodes_with_current = 0
        
        # Definiamo i percorsi delle due cartelle di storico
        self.history_solo = "models/history/solo"
        self.history_self_play = "models/history/self_play"
        
        # Creiamo le cartelle se non esistono
        os.makedirs(self.history_solo, exist_ok=True)
        os.makedirs(self.history_self_play, exist_ok=True)
        
        if self.mode == "self_play":
            self._load_random_opponent()

    def _load_random_opponent(self):
        """Pesca un avversario a caso cercando in entrambi i rami dello storico."""
        models_solo = glob.glob(f"{self.history_solo}/*.zip")
        models_sp = glob.glob(f"{self.history_self_play}/*.zip")
        
        all_candidates = models_solo + models_sp
        
        # Aggiungiamo anche il modello principale corrente se esiste
        if os.path.exists(config.MODEL_NAME + ".zip"):
            all_candidates.append(config.MODEL_NAME + ".zip")
            
        if len(all_candidates) > 0:
            chosen = random.choice(all_candidates)
            self.opponent_model = PPO.load(chosen)
            print(f"🥊 [ARENA] Nuovo avversario caricato: {os.path.basename(chosen)}")
        else:
            print("⚠️ Nessun modello trovato nello storico. L'avversario rimarrà fermo.")

    def reset(self, seed=None, options=None):
        # In Self-Play, cambiamo avversario ogni 10 episodi per varietà
        if self.mode == "self_play":
            self.episodes_with_current += 1
            if self.episodes_with_current >= 10:
                self._load_random_opponent()
                self.episodes_with_current = 0

        obs_dict, info_dict = self.env.reset(seed=seed, options=options)
        self.current_obs = obs_dict
        return obs_dict["boat_0"], info_dict.get("boat_0", {})

    def step(self, action):
        if self.mode == "self_play" and self.opponent_model:
            action_b, _ = self.opponent_model.predict(self.current_obs["boat_1"], deterministic=True)
        else:
            action_b = np.array([0.0, 0.0])

        actions = {"boat_0": action, "boat_1": action_b}
        obs, rewards, terminations, truncations, infos = self.env.step(actions)
        self.current_obs = obs
        
        return obs["boat_0"], rewards["boat_0"], terminations["boat_0"], truncations["boat_0"], infos["boat_0"]

class SaveHistoryCallback(BaseCallback):
    """Salva una copia del modello ogni tot step nella cartella specifica del mode."""
    def __init__(self, save_freq, mode, verbose=0):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.mode = mode
        self.save_path = f"models/history/{self.mode}"

    def _init_callback(self) -> None:
        os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            path = os.path.join(self.save_path, f"model_{self.num_timesteps}_steps")
            self.model.save(path)
            if self.verbose > 0:
                print(f"\n💾 [STORICO {self.mode.upper()}] Salvata copia: {path}")
        return True

class SuccessTrackingCallback(BaseCallback):
    """Tracks success rate over the last 100 episodes."""
    def __init__(self, verbose=0, check_freq=config.CHECK_FREQ):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.episode_successes = []
        self.episode_distances = []
        self.n_episodes = 0
        self.array_target = []

    def _on_step(self) -> bool:
        if 'infos' in self.locals:
            for infos in self.locals['infos']:
                if 'episode' in infos:
                    self.n_episodes += 1
                    final_dist = infos.get('distance_to_target', 999)
                    target_reached = infos.get('indextarget', 0)
                    self.array_target.append(target_reached)
                    self.episode_distances.append(final_dist)

                    success = False
                    if target_reached == 3:
                      success = True
                    self.episode_successes.append(success)

                    if len(self.episode_successes) > 100:
                        self.episode_successes.pop(0)
                        self.episode_distances.pop(0)

        if self.n_calls > 0 and self.n_calls % self.check_freq == 0:
            print(f"\n{'='*70}")
            print(f"📊 Progress at {self.n_calls:,} steps")
            print(f"{'='*70}")

            if len(self.episode_successes) > 0:
                n_recent = len(self.episode_successes)
                n_successes = sum(self.episode_successes)
                success_rate = (n_successes / n_recent) * 100

                print(f"  Episodes completed: {self.n_episodes}")
                print(f"  Last {n_recent} episodes:")
                print(f"    ✓ Successes: {n_successes}/{n_recent} ({success_rate:.1f}%)")
                print(f"    ✗ Failures: {n_recent - n_successes}/{n_recent} ({100-success_rate:.1f}%)")

                avg_dist = np.mean(self.episode_distances)
                min_dist = np.min(self.episode_distances)
                print(f"    Avg final distance: {avg_dist:.1f}m")
                print(f"    Best distance: {min_dist:.1f}m")
                print(f"    📋 Gate History (Last 10): {self.array_target[-10:]}")
            else:
                print(f"  ⏳ No episodes completed yet...")

            print(f"{'='*70}\n")

        return True

    def _on_training_end(self) -> None:
        print(f"\n{'='*70}")
        print(f"🏁 TRAINING COMPLETE!")
        print(f"{'='*70}")

        if len(self.episode_successes) > 0:
            n_recent = min(100, len(self.episode_successes))
            recent_successes = self.episode_successes[-n_recent:]
            n_successes = sum(recent_successes)
            success_rate = (n_successes / n_recent) * 100

            print(f"  Total episodes: {self.n_episodes}")
            print(f"\n  📈 Performance (last {n_recent} episodes):")
            print(f"    ✓ Successes: {n_successes}/{n_recent} ({success_rate:.1f}%)")
            print(f"    ✗ Failures: {n_recent - n_successes}/{n_recent} ({100-success_rate:.1f}%)")

            recent_distances = self.episode_distances[-n_recent:]
            avg_dist = np.mean(recent_distances)
            min_dist = np.min(recent_distances)
            max_dist = np.max(recent_distances)

            print(f"\n  📏 Distance statistics:")
            print(f"    Average: {avg_dist:.1f}m")
            print(f"    Best: {min_dist:.1f}m")
            print(f"    Worst: {max_dist:.1f}m")

            if len(self.episode_successes) >= 100:
                first_50_rate = (sum(self.episode_successes[:50]) / 50) * 100
                last_50_rate = (sum(self.episode_successes[-50:]) / 50) * 100
                improvement = last_50_rate - first_50_rate

                print(f"\n  📊 Learning trend:")
                print(f"    First 50 episodes: {first_50_rate:.1f}% success")
                print(f"    Last 50 episodes: {last_50_rate:.1f}% success")
                print(f"    Improvement: {improvement:+.1f}%")

        print(f"{'='*70}\n")

def training(mode="solo"):
    os.makedirs(config.TENSORBOARD_LOG_DIR, exist_ok=True)
    
    print("="*70)
    print(f"🛥️  SAILING RL - MODE: {mode.upper()}")
    print("="*70)

    train_env = SelfPlayWrapper(mode=mode)
    train_env = Monitor(train_env)

    if os.path.exists(config.MODEL_NAME + ".zip"):
        print("Riprendo l'addestramento del modello esistente...")
        model = PPO.load(config.MODEL_NAME, env=train_env)
    else:
        print("Creo un nuovo modello PPO da zero...")
        model = PPO("MlpPolicy", train_env, verbose=0, tensorboard_log=config.TENSORBOARD_LOG_DIR,
                    learning_rate=config.LEARNING_RATE, n_steps=config.N_STEPS, batch_size=config.BATCH_SIZE)

    # --- IMPOSTAZIONE DINAMICA DELLA FREQUENZA DI SALVATAGGIO ---
    if mode == "solo":
        timesteps = config.TOTAL_TIMESTEPS_SOLO
        freq = 300000  # Salva ogni 300.000 step per la maratona in Solo
    else:
        timesteps = config.TOTAL_TIMESTEPS_SELF_PLAY
        freq = 100000  # Salva ogni 100.000 step per le sessioni brevi in Self-Play

    # Configuriamo le callback
    success_cb = SuccessTrackingCallback(verbose=1, check_freq=config.CHECK_FREQ)
    history_cb = SaveHistoryCallback(save_freq=freq, mode=mode, verbose=1) 
    
    model.learn(total_timesteps=timesteps, callback=CallbackList([success_cb, history_cb]))

    model.save(config.MODEL_NAME)
    model.save(f"models/history/{mode}/model_final_{mode}")
    print(f"\n✅ Modello salvato come '{config.MODEL_NAME}' e nello storico {mode}")

    return model, train_env, success_cb