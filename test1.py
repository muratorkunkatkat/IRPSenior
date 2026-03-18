from data.generator import generate_network_data
from GurobiSolver import VDNSolver

def main():
    print("Generating network data...")
    network_data = generate_network_data(num_dealers=5, num_trucks=10, days=7, num_car_types=2) # actual -> 72, 1000, 30, 5
    
    print("Initializing Gurobi Solver...")
    solver = VDNSolver(network_data)
    
    solver.build_model()
    solver.solve()

if __name__ == "__main__":
    main()