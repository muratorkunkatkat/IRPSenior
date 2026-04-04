import json
from heuristic.heuristic_greedy import GreedyHeuristic

def run_test():
    with open('data/data_exp.json', 'r') as f:
        data = json.load(f)
        
    heuristic = GreedyHeuristic(data)
    
    print("Start\n")
    heuristic.run()
    
    dispatches = heuristic.dispatch_log
    
    # Output Sections Beginning

    # OBJECTIVE VALUE
    total_routing_cost = 0
    for d in dispatches:
        route = d['route']
        for i in range(len(route) - 1):
            total_routing_cost += data['R'][route[i]][route[i+1]]
            
    print(f"\nOBJ. VALUE (total Cost): ${total_routing_cost:,.2f}")
    print(f"TOTAL DISPATCHES: {len(dispatches)}")
    print("-"*100)
    
    # FIRST 30 ROUTES
    print("\nFIRST 30 ROUTES")
    for i, r in enumerate(dispatches[:30]):
        cars_carried = r.get('cars_carried', data['Q'][r['car_type']])
        print(f"Day {r['day']:>2} | Truck {r['truck']:>3} | Car {r['car_type']} | "
              f"Time: {r['time']:>2} days | Carried: {cars_carried:>2} | Route: {r['route']}")
              
    if len(dispatches) > 30:
        print(f"{len(dispatches) - 30} more dispatches.")

    car_types = len(data['C'])

    # DAILY DEMAND SATISFACTION
    print(f"\nDAILY DEMAND SATISFACTION (First 10 Days)")
    print("-"*100)
    
    for t in range(30):
        print(f"\nDay {t}")
        for c in range(car_types):
            # total demand for this car type on this specific day
            day_demand = 0
            for j in data['nodes_no_factory']:
                day_demand += data['F'][j][c][t]
            
            # total cars of this type delivered on this specific day
            day_delivered = 0
            for d in dispatches:
                if d['day'] == t and d['car_type'] == c:
                    day_delivered += d.get('cars_carried', data['Q'][c])
                    
            satisfaction_rate = (day_delivered / day_demand * 100) if day_demand > 0 else 0
            print(f"Car Type {c}: Demanded = {day_demand:>4} | Delivered = {day_delivered:>4} | Satisfaction = {satisfaction_rate:>5.1f}%")

    # UNSOLD CARS AT END OF MONTH
    print(f"\nEND OF MONTH INVENTORY REPORT")
    print("="*60)
    
    total_demanded_all_month = sum(
        data['F'][j][c][t] 
        for j in data['nodes_no_factory'] 
        for c in data['C'] 
        for t in data['T']
    )
    
    total_delivered_all_month = sum(d.get('cars_carried', data['Q'][d['car_type']]) for d in dispatches)
    
    unsold = max(0, total_delivered_all_month - total_demanded_all_month)
    shortfall = max(0, total_demanded_all_month - total_delivered_all_month)
    
    print(f"Total Cars Demanded: {total_demanded_all_month:,}")
    print(f"Total Cars Delivered: {total_delivered_all_month:,}")
    
    if unsold > 0:
        print(f"Unsold Cars (salvage): {unsold:,}")
    else:
        print(f"Unmet Demand (backorder): {shortfall:,}")
    print("="*60 + "\n")

if __name__ == "__main__":
    run_test()