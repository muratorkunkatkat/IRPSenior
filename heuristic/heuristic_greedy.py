class GreedyHeuristic:
    def __init__(self, data):
        self.C = data['C']
        self.K = data['K']
        self.N = data['N']
        self.T = data['T']
        self.N_no_fac = data['nodes_no_factory']
        
        self.A = data['A']
        self.Q = data['Q']
        self.V = data['V']
        self.F = data['F']
        
        # lock variables
        self.truck_locked_until = {k: 0 for k in self.K} # tracks when truck is free
        self.inventory = {i: {c: 0 for c in self.C} for i in self.N_no_fac}
        
        # store output
        self.dispatch_log = []

    def get_critical_dealers(self, c, t):
        """list of critical dealers where inventory - forecasted demand < 0"""
        return [i for i in self.N_no_fac if (self.inventory[i][c] - self.F[i][c][t]) < 0]

    def get_remaining_dealer_capacity(self, i):
        """calculates V_i - (sum of all car types currently at dealer i)"""
        if i == 0:
            return float('inf')
        current_inv = sum(self.inventory[i].values())
        return self.V[i] - current_inv

    def run(self):
        for t in self.T:
            for c in self.C:
                while True:
                    available_trucks = [k for k in self.K if self.truck_locked_until[k] <= t]
                    critical_dealers = self.get_critical_dealers(c, t)
                    
                    if not available_trucks or not critical_dealers:
                        break
                        
                    current = 0
                    k = available_trucks[0]
                    available_capacity = self.Q[c]
                    route = [0]
                    route_time = 0
                    
                    while available_capacity > 0 and len(self.get_critical_dealers(c, t)) > 0:
                        
                        # sorting dealers by travel time from 'current'
                        n_sorted = sorted(self.N_no_fac, key=lambda j: self.A[current][j])
                        
                        moving = False
                        for j in n_sorted:
                            if j in self.get_critical_dealers(c, t):
                                cars_needed = self.F[j][c][t] - self.inventory[j][c]
                                
                                if available_capacity < cars_needed and route[-1] != 0:
                                    prev_node = route[-1]
                                    rem_cap = self.get_remaining_dealer_capacity(prev_node)
                                    
                                    if rem_cap > available_capacity:
                                        self.inventory[prev_node][c] += available_capacity
                                    
                                    available_capacity = 0
                                    break
                                
                                # move to j
                                route_time += self.A[current][j]
                                current = j
                                route.append(current)

                                available_capacity -= cars_needed
                                self.inventory[j][c] += cars_needed
                                moving = True
                                break
                                
                        if not moving:
                            # if we went through all sorted dealers and didnt move, break the loop
                            break
                            
                    # clean up remaining capacity if any since trucks cannot bring cars back to factory
                    if available_capacity != 0 and len(route) > 0:
                        if route[-1] != 0:
                            prev_node = route[-1]
                            rem_cap = self.get_remaining_dealer_capacity(prev_node)
                            if rem_cap > available_capacity:
                                self.inventory[prev_node][c] += available_capacity
                                
                    # return to factory
                    route.append(0)
                    route_time += self.A[current][0]
                    self.truck_locked_until[k] = t + route_time
                    
                    # route log
                    self.dispatch_log.append({
                        'day': t,
                        'truck': k,
                        'car_type': c,
                        'route': route,
                        'time': route_time, 
                        'cars_carried': self.Q[c] - available_capacity
                    })
                    
            # end of day inventory update
            for i in self.N_no_fac:
                for c in self.C:
                    self.inventory[i][c] -= self.F[i][c][t]
                    
        return self.dispatch_log