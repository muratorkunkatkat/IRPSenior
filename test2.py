import os
import random
import matplotlib.pyplot as plt
import numpy as np
from data.generator import generate_network_data
from heuristic.heuristic_greedy import GreedyHeuristic

def adapt_exact_data_for_heuristic(data):
    N_nodes, C_types, days = len(data['N']), len(data['C']), len(data['T'])
    A_list = [[0]*N_nodes for _ in range(N_nodes)]
    for (i, j), val in data['A'].items(): A_list[i][j] = val
    F_list = [[[0]*days for _ in range(C_types)] for _ in range(N_nodes)]
    for (i, c, t), val in data['F'].items(): F_list[i][c][t] = val
    data['A'], data['F'] = A_list, F_list
    return data

def main():
    random.seed(44)
    np.random.seed(44)
    
    raw_data = generate_network_data(num_dealers=10, num_trucks=5, days=30, num_car_types=2)
    coords = raw_data['coords']
    data = adapt_exact_data_for_heuristic(raw_data)
    
    heuristic = GreedyHeuristic(data)
    dispatches = heuristic.run()

    os.makedirs('plots_heuristic', exist_ok=True)
    plt.style.use('dark_background')
    colors = ['orange', 'deepskyblue', 'yellow', 'limegreen', 'mediumorchid']
    car_names = {0: "Egea", 1: "Skudo"}
    
    print("Generating 30 Heuristic plots...")
    for t in data['T']:
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.set_title(f"Day {t} - Heuristic Dispatch Schedule", fontsize=18, fontweight='bold', color='white')
        
        for i in data['N']:
            x, y = coords[i]
            if i == 0:
                ax.scatter(x, y, c='red', s=180, zorder=5, edgecolors='white')
                ax.annotate("Factory", (x, y), textcoords="offset points", xytext=(0,12), ha='center', color='red', fontweight='bold')
            else:
                ax.scatter(x, y, c='white', s=90, zorder=4)
                ax.annotate(str(i), (x, y), textcoords="offset points", xytext=(0,10), ha='center', color='white', fontweight='bold')

        # visualization
        day_dispatches = [d for d in dispatches if d['day'] == t]
        for d in day_dispatches:
            k, c, route, time = d['truck'], d['car_type'], d['route'], d['time']
            truck_color = colors[k % len(colors)]
            car_name = car_names.get(c, "Unknown")
            
            for idx in range(len(route) - 1):
                i, j = route[idx], route[idx+1]
                x1, y1 = coords[i]
                x2, y2 = coords[j]
                ax.annotate("", xy=(x2, y2), xycoords='data', xytext=(x1, y1), textcoords='data',
                            arrowprops=dict(arrowstyle="->", color=truck_color, lw=3.5, shrinkA=6, shrinkB=6), zorder=3)
                
                if i == 0:
                    mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
                    label_text = f"T{k} ({car_name})\n{time} days"
                    ax.text(mid_x, mid_y, label_text, color=truck_color, fontsize=11, fontweight='bold', ha='center', va='bottom', bbox=dict(facecolor='black', alpha=0.75, edgecolor='none', pad=2))

        ax.grid(True, color='#444444', linestyle='--', linewidth=0.5)
        plt.tight_layout()
        plt.savefig(f"plots_heuristic/day_{t:02d}.png", dpi=150)
        plt.close(fig)
        
    print("Successfully generated 30 plots in 'plots_heuristic/'")

if __name__ == "__main__":
    main()