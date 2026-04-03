# GurobiSolver.py
import gurobipy as gp
from gurobipy import GRB

class VDNSolver:
    def __init__(self, data):
        self.d = data
        self.model = gp.Model("Vehicle_Distribution_Network")
        self.epsilon = 0.001

        self.C = self.d['C']
        self.K = self.d['K']
        self.N = self.d['N']
        self.T = self.d['T']
        self.N_no_fac = self.d['nodes_no_factory']

    def build_model(self):
        m = self.model
        d = self.d

        print("Variables")
        # DVs
        self.X = m.addVars(self.K, self.N, self.N, self.T, vtype=GRB.BINARY, name="X")
        self.Y = m.addVars(self.K, self.T, vtype=GRB.BINARY, name="Y")
        self.W = m.addVars(self.K, self.T, vtype=GRB.INTEGER, name="W")
        self.L = m.addVars(self.K, self.T, vtype=GRB.CONTINUOUS, name="L")
        self.Z = m.addVars(self.K, self.C, self.T, vtype=GRB.BINARY, name="Z")
        
        self.J = m.addVars(self.K, self.N, self.N, self.T, vtype=GRB.CONTINUOUS, name="J")
        self.D = m.addVars(self.K, self.N, self.C, self.T, vtype=GRB.CONTINUOUS, name="D")
        self.U = m.addVars(self.K, self.N, self.C, self.T, vtype=GRB.CONTINUOUS, name="U")
        
        self.I = m.addVars(self.N, self.C, self.T, vtype=GRB.CONTINUOUS, name="I")
        self.E = m.addVars(self.N, self.C, self.T, vtype=GRB.CONTINUOUS, name="E")
        self.S = m.addVars(self.N, self.C, self.T, vtype=GRB.CONTINUOUS, name="S")

        self.u = m.addVars(self.K, self.N_no_fac, self.T, vtype=GRB.CONTINUOUS, name="u_mtz")

        for k in self.K:
            for c in self.C:
                for t in self.T:
                    m.addConstr(self.D[k, 0, c, t] == 0)
                    m.addConstr(self.U[k, 0, c, t] == 0)
                    m.addConstr(self.I[0, c, t] == 0)
                    m.addConstr(self.E[0, c, t] == 0)
                    m.addConstr(self.S[0, c, t] == 0)

        # Constraints
        
        # Routing Flow
        for k in self.K:
            for t in self.T:
                m.addConstr(gp.quicksum(self.X[k, 0, j, t] for j in self.N) == self.Y[k, t], name=f"start_{k}_{t}")
                m.addConstr(gp.quicksum(self.X[k, i, 0, t] for i in self.N) == self.Y[k, t], name=f"return_{k}_{t}")
                
                for i in self.N:
                    # Flow Balance
                    m.addConstr(
                        gp.quicksum(self.X[k, j, i, t] for j in self.N) == gp.quicksum(self.X[k, i, j, t] for j in self.N), 
                        name=f"flow_balance_{k}_{i}_{t}"
                    )
                    
                    # Locking Trucks
                    m.addConstr(
                        gp.quicksum(self.D[k, i, c, t] + self.U[k, i, c, t] for c in self.C) <= d['M'] * gp.quicksum(self.X[k, j, i, t] for j in self.N),
                        name=f"action_lock_{k}_{i}_{t}"
                    )
        for k in self.K:
            for t in self.T:
                for i in self.N_no_fac:
                    for j in self.N_no_fac:
                        if i != j:
                            m.addConstr(
                                self.u[k, i, t] - self.u[k, j, t] + len(self.N) * self.X[k, i, j, t] <= len(self.N) - 1,
                                name=f"mtz_{k}_{i}_{j}_{t}"
                            )

        # Truck Capacity and Load Conservation
        for k in self.K:
            for t in self.T:
                m.addConstr(gp.quicksum(self.Z[k, c, t] for c in self.C) <= self.Y[k, t], name=f"single_type_{k}_{t}")
                m.addConstr(gp.quicksum(self.J[k, j, 0, t] for j in self.N) == 0, name=f"empty_return_{k}_{t}")
                
                for i in self.N:
                    for j in self.N:
                        m.addConstr(self.J[k, i, j, t] <= d['M'] * self.X[k, i, j, t])
                        m.addConstr(self.J[k, i, j, t] <= gp.quicksum(d['Q'][c] * self.Z[k, c, t] for c in self.C))
                        
                for i in self.N_no_fac:
                    m.addConstr(
                        gp.quicksum(self.J[k, j, i, t] for j in self.N) + 
                        gp.quicksum(self.U[k, i, c, t] for c in self.C) - 
                        gp.quicksum(self.D[k, i, c, t] for c in self.C) == 
                        gp.quicksum(self.J[k, i, j, t] for j in self.N),
                        name=f"load_balance_{k}_{i}_{t}"
                    )
                    
                    for c in self.C:
                        m.addConstr(self.D[k, i, c, t] + self.U[k, i, c, t] <= d['Q'][c] * self.Z[k, c, t])

        # Route Time Tracking
        for k in self.K:
            for t in self.T:
                m.addConstr(gp.quicksum(d['A'][i, j] * self.X[k, i, j, t] for i in self.N for j in self.N) <= self.W[k, t])
                m.addConstr(self.L[k, t] <= d['M'] * (1 - self.Y[k, t]))
                
                if t > 0:
                    m.addConstr(self.L[k, t] >= self.L[k, t-1] - 1 + self.W[k, t-1])
                else:
                    m.addConstr(self.L[k, t] >= 0)

        # Daily Dealer Inventory
        for i in self.N_no_fac:
            for t in self.T:
                m.addConstr(gp.quicksum(self.I[i, c, t] for c in self.C) <= d['V'][i])
                
                for c in self.C:
                    if t > 0:
                        prev_net = (1 - d['P'][t]) * (self.I[i, c, t-1] - self.E[i, c, t-1])
                        m.addConstr(self.S[i, c, t-1] == d['P'][t] * self.I[i, c, t-1])
                    else:
                        prev_net = 0
                        
                    m.addConstr(
                        self.I[i, c, t] - self.E[i, c, t] == 
                        prev_net + 
                        gp.quicksum(self.D[k, i, c, t] for k in self.K) - 
                        gp.quicksum(self.U[k, i, c, t] for k in self.K) - 
                        d['F'][i, c, t],
                        name=f"inv_flow_{i}_{c}_{t}"
                    )

        # obj. func
        print("setting obj. func")
        obj = (
            gp.quicksum(d['R'][i, j] * self.X[k, i, j, t] for k in self.K for i in self.N for j in self.N for t in self.T) +
            gp.quicksum(d['H'][c] * self.I[i, c, t] for i in self.N_no_fac for c in self.C for t in self.T) +
            gp.quicksum(d['B'][c] * self.E[i, c, t] for i in self.N_no_fac for c in self.C for t in self.T) +
            gp.quicksum(d['G'][c] * self.S[i, c, t] for i in self.N_no_fac for c in self.C for t in self.T) +
            self.epsilon * gp.quicksum(self.W[k, t] for k in self.K for t in self.T)
        )
        m.setObjective(obj, GRB.MINIMIZE)

    def solve(self):
        self.model.setParam('TimeLimit', 600) # 10 minutes max
        self.model.optimize()
        
        if self.model.Status == GRB.OPTIMAL or self.model.Status == GRB.TIME_LIMIT:
            print("\nRESULTS")
            print(f"Obj Value: {self.model.ObjVal}")
            # Quick check on how many cars were unsold
            total_unsold = sum(self.S[i,c,t].X for i in self.N_no_fac for c in self.C for t in self.T)
            print(f"Unsold Cars: {total_unsold}")
        else:
            print("failed to find a feasible solution:", self.model.Status)
            self.model.computeIIS()
            self.model.write("infeasible.ilp")
            print("Wrote infeasibility to 'infeasible.ilp'")
            