from heuristic.heuristic_greedy import GreedyHeuristic

class DynamicGreedyHeuristic(GreedyHeuristic):
    def __init__(self, data):
        super().__init__(data)

    def run(self):
        """
        overrides the base run() to implement Dynamic Queue logic.
        """
        for t in self.T:
            for k in self.K:
                if self.truck_locked_until[k] > t:
                    continue
                
                # assign a car type to this truck based on unmet demand
                best_c = None
                max_unmet = -1
                for c in self.C:
                    unmet = sum(max(0, self.F[i][c][t] - self.inventory[i][c]) for i in self.N_no_fac)
                    if unmet > max_unmet and unmet > 0:
                        max_unmet = unmet
                        best_c = c
                
                if best_c is None:
                    continue
                
                current_node = 0
                route = [0]
                cars_remaining = self.Q[best_c]
                route_time = 0
                
                while cars_remaining > 0:
                    critical_dealers = [i for i in self.N_no_fac if self.inventory[i][best_c] < self.F[i][best_c][t] and i not in route]
                    
                    if not critical_dealers:
                        break
                        
                    next_node = min(critical_dealers, key=lambda x: self.A[current_node][x])
                    
                    route_time += self.A[current_node][next_node]
                    route.append(next_node)
                    
                    needed = self.F[next_node][best_c][t] - self.inventory[next_node][best_c]
                    delivered = min(cars_remaining, needed)
                    
                    self.inventory[next_node][best_c] += delivered
                    cars_remaining -= delivered
                    current_node = next_node
                
                if len(route) > 1:
                    route_time += self.A[current_node][0]
                    route.append(0)
                    self.truck_locked_until[k] = t + route_time
                    
                    self.dispatch_log.append({
                        'day': t,
                        'truck': k,
                        'car_type': best_c,
                        'route': route,
                        'time': route_time,
                        'cars_carried': self.Q[best_c] - cars_remaining
                    })
                    
            for c in self.C:
                for i in self.N_no_fac:
                    self.inventory[i][c] -= self.F[i][c][t]
                    
        return self.dispatch_log