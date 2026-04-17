import imageio
import numpy as np
from stable_baselines3 import PPO
from environment import ImprovedSailingEnv
import config

def generate_video():
    print("\n" + "="*70)
    print("🎬 GENERAZIONE VIDEO CON ANALISI RISULTATI")
    print("="*70)
    
    try:
        model = PPO.load(config.MODEL_NAME)
    except Exception as e:
        print(f"❌ Errore: Modello '{config.MODEL_NAME}' non trovato.")
        return

    env = ImprovedSailingEnv(render_mode='rgb_array')
    max_attempts = config.MAX_VIDEO_ATTEMPTS
    
    results = [] # Lista per memorizzare i dati di ogni tentativo
    winning_seed = None
    best_fallback_seed = 0
    best_fallback_gate = -1
    best_fallback_dist = 9999

    print(f"🔍 ANALISI DI {max_attempts} TENTATIVI IN CORSO...")
    print(f"{'SEED':<8} | {'STATO':<12} | {'GATE':<6} | {'DISTANZA':<10} | {'STEP':<6}")
    print("-" * 60)

    for attempt in range(max_attempts):
        obs, _ = env.reset(seed=attempt)
        env.action_spaces["boat_1"].seed(attempt)
        done = False
        step = 0
        info = {}

        while not done and step < config.MAX_STEPS:
            actions = {}
            for agent_id in env.agents:
                if agent_id == "boat_0":
                    # La barca blu usa il cervello addestrato
                    action, _ = model.predict(obs[agent_id], deterministic=True)
                else:
                    # La barca rossa (boat_1) fa mosse completamente a caso (Dummy)
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
        
        # Determiniamo lo stato per la stampa
        status = "VITTORIA" if (term_b0 and current_gate >= 3) else "FALLITO"
        if trunc_b0: status = "TIMEOUT"

        # Stampiamo il risultato del tentativo corrente
        print(f"{attempt:<8} | {status:<12} | {current_gate:<6} | {current_dist:<10.1f} | {step:<6}")

        # Logica per trovare il migliore (Vittoria > Più Gate > Più Vicino)
        is_winner = (term_b0 and current_gate >= 3)
        if is_winner and winning_seed is None:
            winning_seed = attempt
            
        if current_gate > best_fallback_gate:
            best_fallback_gate = current_gate
            best_fallback_dist = current_dist
            best_fallback_seed = attempt
        elif current_gate == best_fallback_gate:
            if current_dist < best_fallback_dist:
                best_fallback_dist = current_dist
                best_fallback_seed = attempt

    # --- SCELTA DEL SEED DA RENDERIZZARE ---
    seed_to_render = winning_seed if winning_seed is not None else best_fallback_seed
    
    print("-" * 60)
    if winning_seed is not None:
        print(f"✅ Analisi completata. Trovata una vittoria al seed {winning_seed}!")
    else:
        print(f"⚠️ Nessuna vittoria. Miglior risultato: Seed {best_fallback_seed} (Gate: {best_fallback_gate})")
    
    print(f"🎥 Inizio registrazione video per il Seed: {seed_to_render}...")

    # --- REPLAY E REGISTRAZIONE ---
    obs, _ = env.reset(seed=seed_to_render)
    env.action_spaces["boat_1"].seed(attempt)
    frames = [env.render()] 
    done = False
    step = 0

    while not done and step < config.MAX_STEPS:
        actions = {}
        for agent_id in env.agents:
            if agent_id == "boat_0":
                action, _ = model.predict(obs[agent_id], deterministic=True)
            else:
                action = env.action_spaces[agent_id].sample()
            actions[agent_id] = action
            
        obs, rewards, terminations, truncations, infos = env.step(actions)
        frames.append(env.render()) 
        
        term_b0 = terminations.get("boat_0", False)
        trunc_b0 = truncations.get("boat_0", False)
        done = term_b0 or trunc_b0
        step += 1

    for _ in range(15): 
        frames.append(frames[-1])

    print(f"\n💾 Salvataggio in corso: {config.VIDEO_NAME}")
    imageio.mimsave(config.VIDEO_NAME, frames, fps=config.VIDEO_FPS)
    print("✅ Video salvato con successo!")

    env.close()