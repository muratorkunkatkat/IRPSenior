import time
from data.generator import generate_network_data
from GurobiSolver import VDNSolver

def run_backorder_sensitivity():
    # We will multiply the base backorder penalty by these factors
    b_multipliers = [0.5, 1.0, 2.0, 5.0]
    
    print("Running Backorder Penalty Sensitivity (7-Day Exact Model)...")
    print(f"{'B Multiplier':<12} | {'Total Obj':<12} | {'Backorder ($)':<15} | {'Unmet Units':<15} | {'Trans Cost':<12} | {'Time (s)':<8}")
    print("-" * 80)

    for mult in b_multipliers:
        data = generate_network_data(num_dealers=10, num_trucks=6, days=7, num_car_types=2)
        
        # the multiplier to the backorder costs for all car types
        for c in data['C']:
            data['B'][c] = data['B'][c] * mult
            
        solver = VDNSolver(data)
        solver.build_model()
        solver.model.setParam('TimeLimit', 500)
        solver.model.setParam('OutputFlag', 0)
        
        start_time = time.time()
        solver.model.optimize()
        comp_time = time.time() - start_time
        
        if solver.model.Status in [2, 9]: # Optimal or Time Limit Reached
            total_obj = solver.model.ObjVal
            
            # total dollars spent on backorder penalties
            backorder_cost_dollars = sum(data['B'][c] * solver.E[i,c,t].X for i in solver.N_no_fac for c in solver.C for t in solver.T)
            
            # total raw number of unmet units across the 7 days
            total_unmet_units = sum(solver.E[i,c,t].X for i in solver.N_no_fac for c in solver.C for t in solver.T)
            
            # transportation cost for comparison
            trans_cost = sum(data['R'][i,j] * solver.X[k,i,j,t].X for k in solver.K for i in solver.N for j in solver.N for t in solver.T)
            
            print(f"{mult:<12} | {total_obj:<12.2f} | {backorder_cost_dollars:<15.2f} | {total_unmet_units:<15.0f} | {trans_cost:<12.2f} | {comp_time:<8.2f}")
        else:
            print(f"{mult:<12} | Failed to find feasible solution.")

if __name__ == "__main__":
    run_backorder_sensitivity()