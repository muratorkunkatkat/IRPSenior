# data/generator.py
import math
import random

def generate_network_data(num_dealers=5, num_trucks=10, days=7, num_car_types=2): # original: 72, 1000, 30, 5

    random.seed(44)

    N_nodes = num_dealers + 1
    C_types = num_car_types
    
    data = {
        'C': list(range(C_types)),
        'K': list(range(num_trucks)),
        'N': list(range(N_nodes)),
        'T': list(range(days)),
        'nodes_no_factory': list(range(1, N_nodes))
    }

    coords = {i: (random.randint(0, 1000), random.randint(0, 1000)) for i in data['N']}
    
    # A_ij: Travel time and R_ij: Transportation cost
    data['A'] = {}
    data['R'] = {}
    speed_km_per_day = 600 
    cost_per_km = 1.5

    for i in data['N']:
        for j in data['N']:
            if i == j:
                data['A'][i, j] = 0
                data['R'][i, j] = 0.01 # so that xii route becomes eliminated
            else:
                dist = math.hypot(coords[i][0] - coords[j][0], coords[i][1] - coords[j][1])
                # Travel time rounding to int
                data['A'][i, j] = math.ceil(dist / speed_km_per_day)
                data['R'][i, j] = round(dist * cost_per_km, 2)

    # costs
    data['B'] = {c: 500 for c in data['C']}     # backorder cost per day
    data['H'] = {c: 20 for c in data['C']}      # holding cost per day
    data['G'] = {c: 15000 for c in data['C']}   # unsold cost -> very high relatively
    
    # truck capacity
    data['Q'] = {c: random.choice([6, 8, 10]) for c in data['C']}

    # dealer capacity
    data['V'] = {i: random.randint(20, 50) for i in data['nodes_no_factory']}

    # start of month switch
    data['P'] = {t: 1 if t == 0 else 0 for t in data['T']}

    # forecasted demand
    data['F'] = {}
    for i in data['nodes_no_factory']:
        for c in data['C']:
            for t in data['T']:
                # car demands
                data['F'][i, c, t] = random.choices([0, 1, 2, 3], weights=[70, 15, 10, 5])[0]

    data['M'] = 10000 
    data['coords'] = coords

    return data