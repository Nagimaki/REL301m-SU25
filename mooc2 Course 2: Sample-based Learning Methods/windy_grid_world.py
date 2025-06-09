#######################################################################
# Copyright (C)
# 2016-2018 Shangtong Zhang(zhangshangtong.cpp@gmail.com)
# 2016 Kenta Shimada(hyperkentakun@gmail.com)
# Permission given to modify the code as long as you keep this
# declaration at the top
#######################################################################

import numpy as np

# --- Cấu hình môi trường ---
WORLD_HEIGHT = 7
WORLD_WIDTH  = 10
WIND = [0,0,0,1,1,1,2,2,1,0]

ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT = 0,1,2,3
ACTIONS = [ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT]

EPSILON = 0.1
ALPHA   = 0.5
REWARD  = -1.0

START = [3,0]
GOAL  = [3,7]

# --- Hàm step và chọn action ---
def step(state, action):
    i, j = state
    wind = WIND[j]
    if   action == ACTION_UP:    ni, nj = max(i-1-wind, 0), j
    elif action == ACTION_DOWN:  ni, nj = min(max(i+1-wind,0), WORLD_HEIGHT-1), j
    elif action == ACTION_LEFT:  ni, nj = max(i-wind,0), max(j-1,0)
    elif action == ACTION_RIGHT: ni, nj = max(i-wind,0), min(j+1, WORLD_WIDTH-1)
    else: raise ValueError
    return [ni, nj]

def choose_action(q, state):
    # ε-greedy
    if np.random.rand() < EPSILON:
        return np.random.choice(ACTIONS)
    vals = q[state[0], state[1], :]
    maxv = np.max(vals)
    return np.random.choice([a for a,v in enumerate(vals) if v==maxv])

# --- Chạy một episode SARSA ---
def episode(q):
    s = START.copy()
    a = choose_action(q, s)
    while s != GOAL:
        s2 = step(s, a)
        a2 = choose_action(q, s2)
        q[s[0], s[1], a] += ALPHA * (REWARD + q[s2[0], s2[1], a2] - q[s[0], s[1], a])
        s, a = s2, a2

# --- Học SARSA ---
def train_sarsa(n_episodes=500):
    q = np.zeros((WORLD_HEIGHT, WORLD_WIDTH, 4))
    for _ in range(n_episodes):
        episode(q)
    return q

# --- In đường đi của một episode theo greedy policy ---
def print_one_greedy_path(q):
    policy = np.argmax(q, axis=2)
    s = START.copy()
    path = [tuple(s)]
    while s != GOAL:
        a = policy[s[0], s[1]]
        s = step(s, a)
        path.append(tuple(s))
    print("Greedy path from START to GOAL:")
    print(path)

# --- Mô phỏng để tính xác suất đi qua mỗi ô ---
def compute_visit_prob(q, n_runs=5000):
    policy = np.argmax(q, axis=2)
    counts = np.zeros((WORLD_HEIGHT, WORLD_WIDTH))
    for _ in range(n_runs):
        s = START.copy()
        visited = set()
        while s != GOAL:
            visited.add(tuple(s))
            a = policy[s[0], s[1]]
            s = step(s, a)
        visited.add(tuple(GOAL))
        for (i,j) in visited:
            counts[i,j] += 1
    return counts / n_runs

# --- Main ---
if __name__ == '__main__':
    # 1) Train
    q = train_sarsa(500)
    # 2) Show one example path
    print_one_greedy_path(q)
    # 3) Compute and print probability map
    probs = compute_visit_prob(q, 5000)
    print("\nXác suất mỗi ô nằm trên đường đi (7 hàng × 10 cột):")
    print("(giá trị ∈ [0,1])")
    for row in probs:
        print(' '.join(f"{p:.2f}" for p in row))
