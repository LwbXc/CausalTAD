import pdb
import json
import math
import pickle
import os
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc

# Constants
GRID_SIZE = 300  # 300 meters
EARTH_RADIUS = 6371000  # Earth's radius in meters

def haversine(lon1, lat1, lon2, lat2):
    """Calculate the great-circle distance between two points on the Earth."""
    dlon = math.radians(lon2 - lon1)
    dlat = math.radians(lat2 - lat1)
    a = (math.sin(dlat / 2) ** 2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dlon / 2) ** 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return EARTH_RADIUS * c

def get_neighbors(grid):
    """Get the 8 neighboring grids of a given grid."""
    x, y = grid
    return [(x + dx, y + dy) for dx in [-1, 0, 1] for dy in [-1, 0, 1]]

def preprocess_trajectories(train_trajs, config_size):
    """Preprocess the trajectories in the training file."""
    for k,v in train_trajs.items():
        traj = {'o_geo': v['o_geo'], 'grid': list()}
        for g in v['grid']:
            if len(traj['grid'])==0:
                traj['grid'].append(g)
            else:
                next_g = (g//config_size[1], g%config_size[1])
                while True:
                    current_g = (traj['grid'][-1]//config_size[1], traj['grid'][-1]%config_size[1])
                    neighbors = get_neighbors(current_g)
                    if next_g in neighbors:
                        traj['grid'].append(g)
                        break
                    else:
                        min_distance = float('inf')
                        selected_neighbor = None
                        for n in neighbors:
                            cur_distance = abs(n[0]-next_g[0]) + abs(n[1]-next_g[1])
                            if cur_distance<min_distance:
                                min_distance = cur_distance
                                selected_neighbor = n
                        traj['grid'].append(selected_neighbor[0]*config_size[1]+selected_neighbor[1])
        train_trajs[k] = traj

    print("Building inv_index ...")
    sd2trajs = defaultdict(list)
    for k,v in train_trajs.items():
        sd = str([v['grid'][0], v['grid'][-1]])       
        sd2trajs[sd].append(k)

    inv_index = defaultdict(dict)
    with ThreadPoolExecutor() as executor:
        futures = []
        for k,v in train_trajs.items():
            futures.append(executor.submit(inv_index_for_traj, k, v))

        for future in tqdm(as_completed(futures), total=len(futures)):
            tmp_inv_index = future.result()
            for grid, traj2pos in tmp_inv_index.items():
                inv_index[grid] = inv_index[grid] | traj2pos
    
    print("Completed")
    return train_trajs, inv_index, sd2trajs


def inv_index_for_traj(k, traj):
    last_grid = None
    inv_index = defaultdict(dict)
    for pos, grid in enumerate(traj['grid']):
        if grid==last_grid:
            continue
        if k not in inv_index[grid].keys():
            inv_index[grid][k] = pos
        last_grid = grid

    return inv_index

def has_path(T, w, inv_index, config_size):
    """Find trajectories in T that contain the grid sequence w."""
    result = []
    for traj_idx in T:
        positions = []
        for grid in w:
            neighbors = get_neighbors([grid//config_size[1], grid%config_size[1]])
            found = float('inf')
            for neighbor in neighbors:
                neighbor_grid = neighbor[0]*config_size[1] + neighbor[1]
                if traj_idx in inv_index[neighbor_grid].keys():
                    pos = inv_index[neighbor_grid][traj_idx]
                    if pos<found:
                        found = pos
            if not np.isinf(found) and (not positions or found >= positions[-1]):
                positions.append(found)
            else:
                break

        if len(positions) == len(w):
            result.append(traj_idx)
    return result

def calculate_score_for_traj(test_traj, sd2trajs, inv_index, r, config_size):
    sd = str([test_traj['grid'][0], test_traj['grid'][-1]])
    w = []
    score = 0
    if sd not in sd2trajs.keys():
        return 0
    else:
        T = set(sd2trajs[sd])  # All trajectories initially

    prev_point = None

    for coordinate, grid in zip(test_traj['o_geo'][:int(r*len(test_traj['o_geo']))], test_traj['grid'][:int(r*len(test_traj['o_geo']))]):
        if len(w)!=0 and grid==w[-1]:
            continue
        w.append(grid)
        new_T = has_path(T, w, inv_index, config_size)

        if len(T)==0:
            pdb.set_trace()
        theta = len(new_T) / len(T)

        if theta < 0.05:
            T = set(sd2trajs[sd])
            w = [grid]
        else:
            T = new_T

        if prev_point:
            dist = haversine(prev_point[0], prev_point[1], coordinate[0], coordinate[1])
            sigma = 1 / (1 + math.exp(150 * (theta - 0.05)))
            score += dist * sigma

        prev_point = coordinate

    return score

def calculate_score(test_trajectories, inv_index, sd2trajs, r, config_size):
    """Calculate the anomaly score for each trajectory in the test file."""

    scores = list()
    with ThreadPoolExecutor() as executor:
        futures = []
        for test_idx, test_traj in test_trajectories.items():
            futures.append(executor.submit(calculate_score_for_traj, test_traj, sd2trajs, inv_index, r, config_size))

        for future in tqdm(as_completed(futures), total=len(futures)):
            score = future.result()
            scores.append(score)        

    return np.array(scores)

def main(city):
    normal_trajs = pickle.load(open(f'../datasets/{city}/test-grid.pkl', 'rb'))
    config = json.load(open(f'../datasets/{city}/config.json'))

    if not os.path.exists(f"./inv_index/{city}/inv_index.pkl"):
        # Preprocess the training trajectories
        history_trajs = pickle.load(open(f'../datasets/{city}/train-grid.pkl', 'rb'))
        grid_trajectories, inv_index, sd2trajs = preprocess_trajectories(history_trajs, config['grid_size'])
        pickle.dump(inv_index, open(f"./inv_index/{city}/inv_index.pkl", 'wb'))
        pickle.dump(sd2trajs, open(f"./inv_index/{city}/sd2trajs.pkl", 'wb'))
    else:
        if not os.path.exists(f"./inv_index/{city}/"):
            os.makedirs(f"./inv_index/{city}/")

        inv_index = pickle.load(open(f"./inv_index/{city}/inv_index.pkl", 'rb'))
        sd2trajs = pickle.load(open(f"./inv_index/{city}/sd2trajs.pkl", 'rb'))


    pr_auc_list = dict()
    for r in [0.5, 0.7, 1.0]:
        normal_scores = calculate_score(normal_trajs, inv_index, sd2trajs, r, config['grid_size'])
        for a in [0.1, 0.2, 0.3]:
            for d in [1, 2, 3]:
                abnormal_trajs = pickle.load(open(f'../datasets/{city}/alpha_{a}_distance_{d}.pkl', 'rb'))
                abnormal_scores = calculate_score(abnormal_trajs, inv_index, sd2trajs, r, config['grid_size'])

                score = np.concatenate((abnormal_scores, normal_scores))
                label = np.concatenate((np.ones(len(abnormal_scores)), np.zeros(len(normal_scores))))
                pre, rec, _t = precision_recall_curve(label, score)
                pr = round(auc(rec, pre), 4)
                print(f"a_{a}_d_{d}_r_{r}: {pr}")
    
                pr_auc_list.setdefault(f"a_{a}_d_{d}_r_{r}", list())
                pr_auc_list[f"a_{a}_d_{d}_r_{r}"].append(pr)

    with open("log.txt", 'a') as f:
        f.write(f'{city}:\n')

    post = '|iBOAT'
    for a in [0.1, 0.2, 0.3]:
        for d in [1, 2, 3]:
            for r in [0.5, 0.7, 1.0]:
                post += f'|{pr_auc_list[f"a_{a}_d_{d}_r_{r}"][-1]}'

    print(post)
    with open("log.txt", 'a') as f:
        f.write(post + '\n')

if __name__ == "__main__":
    main('chengdu')
    main('porto')