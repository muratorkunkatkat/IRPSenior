import os
import random
import matplotlib.pyplot as plt
import numpy as np
from data.generator import generate_network_data
from GurobiSolver import VDNSolver

def main():
    # seeds for reproducibility
    random.seed(44)
    np.random.seed(44)
    
    # 1 month, 5 trucks, 2 car types
    print("Network data generalization")
    network_data = generate_network_data(num_dealers=10, num_trucks=5, days=30, num_car_types=2)
    coords = network_data['coords']
    
    solver = VDNSolver(network_data)
    solver.build_model()
    
    solver.model.setParam('TimeLimit', 600) 
    solver.solve()

    if solver.model.Status not in [2, 9]: 
        print("No feasible solution to plot.")
        return

    # visualization
    os.makedirs('plots', exist_ok=True)
    plt.style.use('dark_background')
    colors = ['orange', 'deepskyblue', 'yellow', 'limegreen', 'mediumorchid']
    
    # car mappings
    car_names = {0: "Egea", 1: "Skudo"}
    
    T = solver.T
    K = solver.K
    N = solver.N
    
    print("\30 plots")
    for t in T:
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.set_title(f"Day {t} - Dispatch Schedule", fontsize=18, fontweight='bold', color='white')
        ax.set_xlabel("X Coordinate", fontweight='bold')
        ax.set_ylabel("Y Coordinate", fontweight='bold')
        
        # ploting all dealers and factory (red)
        for i in N:
            x, y = coords[i]
            if i == 0:
                ax.scatter(x, y, c='red', s=180, zorder=5, edgecolors='white', label='Factory (0)')
                ax.annotate("Factory", (x, y), textcoords="offset points", xytext=(0,12), 
                            ha='center', color='red', fontweight='bold')
            else:
                ax.scatter(x, y, c='white', s=90, zorder=4)
                ax.annotate(str(i), (x, y), textcoords="offset points", xytext=(0,10), 
                            ha='center', color='white', fontsize=10, fontweight='bold')

        # plotting routes for each truck dispatched on day t
        for k in K:
            # if truck k starts a tour today
            if solver.Y[k, t].X > 0.5:
                truck_color = colors[k % len(colors)]
                
                # which car type the truck is carrying
                car_type = None
                for c in solver.C:
                    if solver.Z[k, c, t].X > 0.5:
                        car_type = c
                        break
                
                car_name = car_names.get(car_type, "Unknown")
                
                # tour time
                tour_time = max(0, int(round(solver.W[k, t].X)))
                
                # route plotting
                for i in N:
                    for j in N:
                        if solver.X[k, i, j, t].X > 0.5:
                            x1, y1 = coords[i]
                            x2, y2 = coords[j]
                            
                            ax.annotate("",
                                        xy=(x2, y2), xycoords='data',
                                        xytext=(x1, y1), textcoords='data',
                                        arrowprops=dict(arrowstyle="->", color=truck_color, lw=3.5, shrinkA=6, shrinkB=6),
                                        zorder=3)
                            
                            if i == 0:
                                mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
                                label_text = f"T{k} ({car_name})\n{tour_time} days"
                                ax.text(mid_x, mid_y, label_text, color=truck_color, 
                                        fontsize=11, fontweight='bold', ha='center', va='bottom',
                                        bbox=dict(facecolor='black', alpha=0.75, edgecolor='none', pad=2))

        # legends
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        if by_label:
            ax.legend(by_label.values(), by_label.keys(), loc='upper left', prop={'weight':'bold'})

        # clean & save
        ax.grid(True, color='#444444', linestyle='--', linewidth=0.5)
        plt.tight_layout()
        plt.savefig(f"plots/day_{t:02d}.png", dpi=150)
        plt.close(fig)
        
    print("Successfully generated 30 plots in the 'plots/' directory!")

if __name__ == "__main__":
    main()