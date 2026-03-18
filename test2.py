from data.generator import generate_network_data
from GurobiSolver import VDNSolver

def main():
    print("Network data generalization")
    # 10 dealers, 25 trucks, 7 days, 2 car types
    # 147000 variables
    network_data = generate_network_data(num_dealers=10, num_trucks=25, days=7, num_car_types=2)
    
    print("GurobiSolver init")
    solver = VDNSolver(network_data)
    
    solver.build_model()
    solver.solve()

    # schedule visualization
    if solver.model.Status == 2: # GRB.OPTIMAL
        print("\nDispatch Schedule Example")
        printed_routes = 0
        for t in solver.T:
            for k in solver.K:
                # if truck k starts a tour on day t
                if solver.Y[k, t].X > 0.5:
                    print(f"Day {t}, truck {k} dispatched")
                    # where it goes
                    for i in solver.N:
                        for j in solver.N:
                            if solver.X[k, i, j, t].X > 0.5:
                                # how many cars is it carrying on i to j
                                load = sum(solver.J[k, i, j, t].X for _ in [1]) 
                                print(f"  -> Drives: from {i}, to {j} (carrying {load} cars)")
                    printed_routes += 1
                    if printed_routes >= 5: # first 5 nodes
                        print("...")
                        return

if __name__ == "__main__":
    main()
    