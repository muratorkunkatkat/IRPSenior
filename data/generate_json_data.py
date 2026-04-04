import json
import math
import random
import os

def generate_json():
    random.seed(42)

    N_nodes = 72
    num_dealers = N_nodes - 1
    num_trucks = 500
    days = 30
    num_car_types = 5

    data = {
        'C': list(range(num_car_types)),
        'K': list(range(num_trucks)),
        'N': list(range(N_nodes)),
        'T': list(range(days)),
        'nodes_no_factory': list(range(1, N_nodes))
    }

    # random coordinates
    coords = {i: (random.randint(0, 1000), random.randint(0, 1000)) for i in data['N']}
    
    data['A'] = [[0] * N_nodes for _ in range(N_nodes)]
    data['R'] = [[0.0] * N_nodes for _ in range(N_nodes)]
    
    speed_km_per_day = 400 
    cost_per_km = 1.5

    for i in data['N']:
        for j in data['N']:
            if i == j:
                data['A'][i][j] = 0
                data['R'][i][j] = 0.0
            else:
                dist = math.hypot(coords[i][0] - coords[j][0], coords[i][1] - coords[j][1])
                data['A'][i][j] = math.ceil(dist / speed_km_per_day)
                data['R'][i][j] = round(dist * cost_per_km, 2)

    # truck capacities
    data['Q'] = [random.choice([10, 15, 20, 25]) for _ in data['C']]
    
    # dealer capacities
    data['V'] = [0] + [random.randint(2000, 3000) for _ in range(num_dealers)]
    
    data['B'] = [500] * num_car_types
    data['H'] = [20] * num_car_types
    data['G'] = [15000] * num_car_types
    data['P'] = [1 if t == 0 else 0 for t in data['T']]

    # car demand
    data['F'] = []
    for i in data['N']:
        node_demand = []
        for c in data['C']:
            car_demand = []
            for t in data['T']:
                if i == 0:
                    car_demand.append(0) #factory demand = 0
                else:
                    car_demand.append(random.randint(2, 10))
            node_demand.append(car_demand)
        data['F'].append(node_demand)

    os.makedirs('data', exist_ok=True)
    filepath = 'data/data_exp.json'
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=4)
        
    print(f"data_exp.json -> nodes: {N_nodes}, trucks: {num_trucks}, cars: {num_car_types}, days: {days}")

if __name__ == "__main__":
    generate_json()