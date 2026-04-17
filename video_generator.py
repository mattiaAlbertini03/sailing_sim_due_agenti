import os
import glob
import imageio
import numpy as np
from stable_baselines3 import PPO
from environment import ImprovedSailingEnv
import config

def generate_video(mode="solo"):
    print("\n" + "="*70)
    print(f"🎬 GENERAZIONE VIDEO ({mode.upper()}) CON ANALISI")
    print("="*70)
    
    try:
        model = PPO.load(config.MODEL_NAME)
    except Exception as e:
        print(f"❌ Errore: Modello '{config.MODEL_NAME}' non trovato.")
        return

    # --- CARICAMENTO AVVERSARIO (Il Modello Precedente) ---
    opponent_model = None
    if mode == "self_play":
        history_sp = glob.glob("models/history/self_play/*.zip")
        history_solo = glob.glob("models/history/solo/*.zip")
        all_models = history_sp + history_solo
        
        if len(all_models) > 0:
            # Ordiniamo i modelli storici per data di modifica
            all_models.sort(key=os.path.getmtime)
            
            # Prendiamo il penultimo (che rappresenta "quello precedente" in ordine di tempo)
            # Se ce n'è solo uno disponibile, prendiamo quello.
            opponent_path = all_models[-2] if len(all_models) > 1 else all_models[0]
            
            print(f"🥊 Avversario caricato per il video: {os.path.basename(opponent_path)}")
            opponent_model = PPO.load(opponent_path)
        else:
            print("⚠️ Nessun modello storico trovato. L'avversario farà mosse casuali.")

    # Passiamo il mode all'ambiente
    env = ImprovedSailingEnv(render_mode='rgb_array', mode=mode)
    max_attempts = config.MAX_VIDEO_ATTEMPTS
    
    winning_seed = None
    best_fallback_seed = 0
    best_fallback_gate = -1
    best_fallback_dist = 9999

    print(f"🔍 ANALISI DI {max_attempts} TENTATIVI IN CORSO...")
    print(f"{'SEED':<8} | {'STATO':<12} | {'GATE':<6} | {'DISTANZA':<10} | {'STEP':<6}")
    print("-" * 60)

    for attempt in range(max_attempts):
        obs, _ = env.reset(seed=attempt)
        if mode == "self_play":
            env.action_spaces["boat_1"].seed(attempt)
            
        done = False
        step = 0
        info = {}

        while not done and step < config.MAX_STEPS:
            actions = {}
            for agent_id in env.agents:
                if agent_id == "boat_0":
                    # Il "Campione in carica" guida la barca 0
                    action, _ = model.predict(obs[agent_id], deterministic=True)
                elif agent_id == "boat_1" and opponent_model is not None:
                    # Il "Modello Precedente" guida la barca 1!
                    action, _ = opponent_model.predict(obs[agent_id], deterministic=True)
                else:
                    action = env.action_spaces[agent_id].sample()
                actions[agent_id] = action
                
            obs, rewards, terminations, truncations, infos = env.step(actions)
            info = infos.get("boat_0", {})
            
            term_b0 = terminations.get("boat_0", False)
            trunc_b0 = truncations.get("boat_0", False)
            done = term_b0 or trunc_b0
            step += 1

        current_gate = info.get('indextarget', 0)
        current_dist = info.get('distance_to_target', 999)
        
        status = "VITTORIA" if (term_b0 and current_gate >= 3) else "FALLITO"
        if trunc_b0: status = "TIMEOUT"

        print(f"{attempt:<8} | {status:<12} | {current_gate:<6} | {current_dist:<10.1f} | {step:<6}")
        
        is_winner = (term_b0 and current_gate >= 3)
        if is_winner and winning_seed is None: 
            winning_seed = attempt
            
        if current_gate > best_fallback_gate:
            best_fallback_gate = current_gate
            best_fallback_dist = current_dist
            best_fallback_seed = attempt
        elif current_gate == best_fallback_gate and current_dist < best_fallback_dist:
            best_fallback_dist = current_dist
            best_fallback_seed = attempt

    seed_to_render = winning_seed if winning_seed is not None else best_fallback_seed
    
    print("-" * 60)
    if winning_seed is not None:
        print(f"✅ Analisi completata. Trovata una vittoria al seed {winning_seed}!")
    else:
        print(f"⚠️ Nessuna vittoria. Miglior risultato: Seed {best_fallback_seed} (Gate: {best_fallback_gate})")
    
    print(f"🎥 Inizio registrazione video per il Seed: {seed_to_render}...")

    # --- FASE DI RENDERING DEFINITIVO ---
    obs, _ = env.reset(seed=seed_to_render)
    if mode == "self_play": 
        env.action_spaces["boat_1"].seed(seed_to_render)
    frames = [env.render()] 
    done = False
    step = 0

    while not done and step < config.MAX_STEPS:
        actions = {}
        for agent_id in env.agents:
            if agent_id == "boat_0":
                action, _ = model.predict(obs[agent_id], deterministic=True)
            elif agent_id == "boat_1" and opponent_model is not None:
                # Anche nel rendering l'avversario usa l'intelligenza artificiale
                action, _ = opponent_model.predict(obs[agent_id], deterministic=True)
            else:
                action = env.action_spaces[agent_id].sample()
            actions[agent_id] = action
            
        obs, rewards, terminations, truncations, infos = env.step(actions)
        frames.append(env.render()) 
        done = terminations.get("boat_0", False) or truncations.get("boat_0", False)
        step += 1

    for _ in range(15): frames.append(frames[-1])
    imageio.mimsave(config.VIDEO_NAME, frames, fps=config.VIDEO_FPS)
    print("✅ Video salvato con successo!")
    env.close()