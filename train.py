import os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
import gymnasium as gym
# Importiamo l'ambiente e le configurazioni
from environment import ImprovedSailingEnv
import config


class SelfPlayWrapper(gym.Env):
    def __init__(self):
        self.env = ImprovedSailingEnv()
        # Spazi di azione e osservazione per la singola barca
        self.action_space = self.env.action_spaces["boat_0"]

        self.observation_space = self.env.observation_spaces["boat_0"]

        opponent_model_path = config.MODEL_NAME
        # Carichiamo il "fantasma" dell'avversario
        self.opponent_model = None

        #provo a fare un allenamento singolo inizialmente
        if os.path.exists(opponent_model_path + ".zip"):
            print("Avversario caricato: L'agente combatterà contro una versione precedente di se stesso!")
            self.opponent_model = PPO.load(opponent_model_path)
        else:
            print("Nessun modello avversario trovato. L'avversario farà mosse casuali.")

    def reset(self, seed=None, options=None):
        obs_dict, info_dict = self.env.reset(seed=seed, options=options)
        self.current_obs = obs_dict
        return obs_dict["boat_0"], info_dict.get("boat_0", {})

    def step(self, action):
        # 1. Decidiamo l'azione dell'avversario (boat_1)
        if self.opponent_model:
            action_b, _ = self.opponent_model.predict(self.current_obs["boat_1"], deterministic=True)
        else:
            action_b = self.action_space.sample() # Azione casuale se non c'è modello

        # 2. Assembliamo il dizionario delle azioni
        actions = {"boat_0": action, "boat_1": action_b}

        # 3. Facciamo avanzare l'ambiente Multi-Agente
        obs, rewards, terminated, truncated, infos = self.env.step(actions)
        self.current_obs = obs

       # 4. Estraiamo solo i risultati della barca in addestramento (boat_0)
        obs_b0 = obs.get("boat_0", np.zeros(shape=(13,), dtype=np.float32))
        reward = rewards.get("boat_0", 0.0)
        info = infos.get("boat_0", {})
        
        # --- MODIFICA QUI: Estraiamo i booleani dai dizionari ---
        term_b0 = terminated.get("boat_0", False)
        trunc_b0 = truncated.get("boat_0", False)
        
        return obs_b0, reward, term_b0, trunc_b0, info


class SuccessTrackingCallback(BaseCallback):
    """
    Tracks success rate over the last 100 episodes.
    """
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


def training():
    """Main training function"""
    os.makedirs(config.TENSORBOARD_LOG_DIR, exist_ok=True)

    print("="*70)
    print("🛥️  SAILING RL - IMPROVED CONVERGENCE VERSION")
    print("="*70)

    print("1. Creating environment...")
    train_env = SelfPlayWrapper()
    train_env = Monitor(train_env)
    print("Environment created!")

    print("\n2. Creating PPO model with improved hyperparameters...")

    model_path = config.MODEL_NAME
    if os.path.exists(model_path + ".zip"):
        print("Riprendo l'addestramento del modello esistente...")
        model = PPO.load(model_path, env=train_env)
    else:
        print("Creo un nuovo modello PPO da zero...")
        model = PPO(
            "MlpPolicy",
            train_env,
            learning_rate=config.LEARNING_RATE,
            n_steps=config.N_STEPS,
            batch_size=config.BATCH_SIZE,
            n_epochs=config.N_EPOCHS,
            gamma=config.GAMMA,
            gae_lambda=config.GAE_LAMBDA,
            clip_range=config.CLIP_RANGE,
            ent_coef=config.ENT_COEF,
            verbose=0,
            tensorboard_log=config.TENSORBOARD_LOG_DIR
        )
    print("PPO model created!")

    print("\n3. Setting up callback...")
    callback = SuccessTrackingCallback(verbose=1, check_freq=config.CHECK_FREQ)
    print("Callback ready!")

    print(f"\n4. Training for {config.TOTAL_TIMESTEPS} steps...")

    model.learn(
        total_timesteps=config.TOTAL_TIMESTEPS,
        callback=callback,
        progress_bar=False
    )

    model.save(config.MODEL_NAME)
    print(f"\n Model saved as '{config.MODEL_NAME}'")
#TODO chiudere env??
    return model, train_env, callback
