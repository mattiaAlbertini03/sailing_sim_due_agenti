import numpy as np
import matplotlib
matplotlib.use('Agg') #forza il rendering senza display (utile per server o ambienti headless)
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import LineCollection
from utils import normalize_angle

def disegna_boe(env, ax, dx, dy, i):
    # Usiamo env.gates invece di self.gates
    puntoA = (env.gates[i][0] - dx, env.gates[i][1] - dy)
    puntoB = (env.gates[i][0] + dx, env.gates[i][1] + dy)
    
    # Disegna la linea del Gate - rimosso self.ax
    ax.plot([puntoA[0], puntoB[0]], [puntoA[1], puntoB[1]],
            color='orange', linestyle='--', linewidth=2, alpha=0.6)

    # Disegna le due Boe
    boa_a = plt.Circle(puntoA, 5, color='red')
    boa_b = plt.Circle(puntoB, 5, color='red')
    ax.add_patch(boa_a)
    ax.add_patch(boa_b)

def render_frame(env):
    """
    Disegna l'intero frame di gioco prendendo i dati dall'ambiente (env).
    """
    plt.close('all')
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Corretti self.field_size in env.field_size_x/y
    ax.set_xlim(0, env.field_size_x)
    ax.set_ylim(0, env.field_size_y)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    # 1. Linea di partenza/arrivo
    ax.axhline(y=env.finish_line_y, color='black', linestyle='--', linewidth=3, alpha=0.7)
    ax.text(env.field_size_x / 2, env.finish_line_y + 10, "START/FINISH LINE",
                color='black', fontweight='bold', ha='center', fontsize=12)

    #perp_angle = env.start_wind_dir + (np.pi / 2)
    #dx_gate = env.target_radius * np.cos(perp_angle)
    #dy_gate = env.target_radius * np.sin(perp_angle)

    #boe statiche orizzontali
    dx_gate = env.target_radius
    dy_gate = 0.0

    # Chiamata corretta senza self
    disegna_boe(env, ax, dx_gate, dy_gate, 0)
    disegna_boe(env, ax, dx_gate, dy_gate, 1)

    title_text = f"Step: {env.step_count}"

    colors = {"boat_0": "blue", "boat_1": "red"}
    foil_down_colors = {"boat_0": "blue", "boat_1": "red"}
    foil_up_colors = {"boat_0": "cyan", "boat_1": "magenta"}
    arrow_x_value = {"boat_0": 50, "boat_1": env.field_size_x - 50}
    
    boat_size = 20
    boat_points = np.array([ [boat_size, 0],
        [-boat_size/2, boat_size/2],
        [-boat_size/2, -boat_size/2]
    ])

    for agent_id, boat_obj in env.boats.items():
        # Corretta l'assegnazione x, y e i nomi delle variabili
        x, y = boat_obj.x, boat_obj.y
        head = boat_obj.heading
        trajectory = boat_obj.trajectory
        speed = boat_obj.speed
        foil = boat_obj.foil
        id_target = boat_obj.gate_index
        wind_dir = boat_obj.wind_dir
        wind_speed = boat_obj.wind_speed
        
        # Gestione target (se ha finito i gate punta alla linea d'arrivo)
        if id_target < len(env.gates):
            target = env.gates[id_target]
        else:
            target = np.array([x, env.finish_line_y])
            
        dist = np.linalg.norm(np.array([x, y]) - target)
        rel_angle = normalize_angle(wind_dir - head)
        rel_deg = np.degrees(rel_angle)
        
        boat_color = 'gray' if speed < 0.5 else colors[agent_id]
        rot_mat = np.array([[np.cos(head), -np.sin(head)], [np.sin(head), np.cos(head)]])
        boat_draw_pos = boat_points @ rot_mat.T + np.array([x, y])
        
        poly = Polygon(boat_draw_pos, closed=True, color=boat_color, ec='black', lw=1)
        ax.add_patch(poly)
        
        # Traiettoria
        if len(trajectory) > 1:
            traj_data = np.array(trajectory)
            pts = traj_data[:, :2].reshape(-1, 1, 2)
            segments = np.concatenate([pts[:-1], pts[1:]], axis=1)
            foil_states = traj_data[1:, 2]
            # Usiamo un nome diverso per non sovrascrivere il dizionario colors
            line_colors = [foil_up_colors[agent_id] if f > 0.5 else foil_down_colors[agent_id] for f in foil_states]
            lc = LineCollection(segments, colors=line_colors, linewidths=2, alpha=0.6)
            ax.add_collection(lc)
        
        # Frecce del vento
        arrow_x, arrow_y = arrow_x_value[agent_id], 350
        v_dx = 30 * np.cos(wind_dir)
        v_dy = 30 * np.sin(wind_dir)
        ax.arrow(arrow_x, arrow_y, v_dx, v_dy, head_width=10, head_length=10, fc='blue', ec='blue')
        ax.text(arrow_x, arrow_y + 40, f"{wind_speed:.1f} kts", color='blue', ha='center', fontweight='bold')

        title_text += f"\n{agent_id} - Gate: {id_target} | Spd: {speed:.1f} | Dist: {dist:.0f}m | Wind Angle: {rel_deg:.0f}° | Foil: {foil}"

    ax.set_title(title_text, fontsize=10)
    
    fig.canvas.draw()
    image = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    fig.clf()
    plt.close(fig)

    import gc; gc.collect()  # Forza la pulizia della memoria

    return image