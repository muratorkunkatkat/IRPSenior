import time
from data.generator import generate_network_data
from GurobiSolver import VDNSolver

def run_exact_sensitivity():
    truck_scenarios = [4, 6, 8]
    print(f"{'Trucks':<8} | {'Total Obj':<12} | {'Trans Cost':<12} | {'Hold Cost':<12} | {'Backorder':<12} | {'Unsold Cost':<12} | {'Vars':<8} | {'Time (s)':<8}")
    print("-" * 100)

    for trucks in truck_scenarios:
        # 7 days, 2 car types, 10 dealers
        data = generate_network_data(num_dealers=10, num_trucks=trucks, days=30, num_car_types=2)
        
        solver = VDNSolver(data)
        solver.build_model()
        solver.model.setParam('TimeLimit', 600)
        solver.model.setParam('OutputFlag', 0)
        
        start_time = time.time()
        solver.model.optimize()
        comp_time = time.time() - start_time
        
        if solver.model.Status in [2, 9]:
            trans_cost = sum(data['R'][i,j] * solver.X[k,i,j,t].X for k in solver.K for i in solver.N for j in solver.N for t in solver.T)
            hold_cost = sum(data['H'][c] * solver.I[i,c,t].X for i in solver.N_no_fac for c in solver.C for t in solver.T)
            back_cost = sum(data['B'][c] * solver.E[i,c,t].X for i in solver.N_no_fac for c in solver.C for t in solver.T)
            unsold_cost = sum(data['G'][c] * solver.S[i,c,t].X for i in solver.N_no_fac for c in solver.C for t in solver.T)
            
            total_obj = solver.model.ObjVal
            num_vars = solver.model.NumVars
            
            print(f"{trucks:<8} | {total_obj:<12.2f} | {trans_cost:<12.2f} | {hold_cost:<12.2f} | {back_cost:<12.2f} | {unsold_cost:<12.2f} | {num_vars:<8} | {comp_time:<8.2f}")
        else:
            print(f"{trucks:<8}: Failed to find feasible solution within time limit.")

if __name__ == "__main__":
    run_exact_sensitivity()