#######################################################################
# Copyright (C)                                                       #
# 2016-2018 Shangtong Zhang(zhangshangtong.cpp@gmail.com)             #
# 2016 Kenta Shimada(hyperkentakun@gmail.com)                         #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.table import Table

matplotlib.use('Agg')

WORLD_SIZE = 4
# left, up, right, down
ACTIONS = [np.array([0, -1]),
           np.array([-1, 0]),
           np.array([0, 1]),
           np.array([1, 0])]
ACTION_PROB = 0.25


def is_terminal(state):
    x, y = state
    return (x == 0 and y == 0) or (x == WORLD_SIZE - 1 and y == WORLD_SIZE - 1)


def step(state, action):
    if is_terminal(state):
        return state, 0

    next_state = (np.array(state) + action).tolist()
    x, y = next_state

    if x < 0 or x >= WORLD_SIZE or y < 0 or y >= WORLD_SIZE:
        next_state = state

    reward = -1
    return next_state, reward


def draw_image(image):
    fig, ax = plt.subplots()
    ax.set_axis_off()
    tb = Table(ax, bbox=[0, 0, 1, 1])

    nrows, ncols = image.shape
    width, height = 1.0 / ncols, 1.0 / nrows

    # Add cells
    for (i, j), val in np.ndenumerate(image):
        tb.add_cell(i, j, width, height, text=val,
                    loc='center', facecolor='white')

    # Row and column labels
    for i in range(len(image)):
        tb.add_cell(i, -1, width, height, text=i+1, loc='right',
                    edgecolor='none', facecolor='none')
        tb.add_cell(-1, i, width, height/2, text=i+1, loc='center',
                    edgecolor='none', facecolor='none')
    ax.add_table(tb)


def compute_state_value(in_place=True, discount=1.0):
    new_state_values = np.zeros((WORLD_SIZE, WORLD_SIZE))
    iteration = 0
    while True:
        if in_place:
            state_values = new_state_values
        else:
            state_values = new_state_values.copy()
        old_state_values = state_values.copy()

        for i in range(WORLD_SIZE):
            for j in range(WORLD_SIZE):
                value = 0
                for action in ACTIONS:
                    (next_i, next_j), reward = step([i, j], action)
                    value += ACTION_PROB * (reward + discount * state_values[next_i, next_j])
                new_state_values[i, j] = value

        max_delta_value = abs(old_state_values - new_state_values).max()
        if max_delta_value < 1e-4:
            break

        iteration += 1

    return new_state_values, iteration


def figure_4_1():
    # Original Example 4.1
    _, async_iter = compute_state_value(in_place=True)
    values, sync_iter = compute_state_value(in_place=False)
    draw_image(np.round(values, decimals=2))
    print('Figure 4.1:')
    print(' In-place iterations:', async_iter)
    print(' Synchronous iterations:', sync_iter)
    plt.savefig('figure_4_1.png')
    plt.close()


def solve_exercise_4_2(values):
    """
    Compute v_pi(15) for:
      a) new state under 13, but original dynamics unchanged.
      b) in addition, state 13's down action leads to new state.
    Returns: (v15_a, v13_b, v15_b)
    """
    # Extract needed original values
    v12 = values[3, 0]   # state 12 at (3,0)
    v13 = values[3, 1]   # state 13 at (3,1)
    v14 = values[3, 2]   # state 14 at (3,2)
    v9  = values[2, 1]   # neighbor above state13 for part b

    # Part (a):  actions from new state 15 go to 12, 13, 14, and itself
    # v15 = ( (-1+v12) + (-1+v13) + (-1+v14) + (-1+v15) ) / 4
    # => 4 v15 = -4 + v12 + v13 + v14 + v15  => 3 v15 = -4 + v12 + v13 + v14
    v15_a = (v12 + v13 + v14 - 4) / 3.0

    # Part (b): state 13's down now goes to new state 15
    # Setup linear system:
    #   v13 = ( (-1+v12) + (-1+v9) + (-1+v14) + (-1+v15) ) / 4
    #   v15 = ( (-1+v12) + (-1+v13) + (-1+v14) + (-1+v15) ) / 4  but original actions for 15 unchanged
    # Simplify to:
    #   4 v13 - v15 = -4 + v12 + v9 + v14
    #   -1 v13 + 3 v15 = -4 + v12 + v14
    A = np.array([[4.0, -1.0], [-1.0, 3.0]])
    b_vec = np.array([(-4 + v12 + v9 + v14), (-4 + v12 + v14)])
    v13_b, v15_b = np.linalg.solve(A, b_vec)

    return v15_a, v13_b, v15_b


if __name__ == '__main__':
    # Example 4.1
    figure_4_1()

    # Exercise 4.2
    values, _ = compute_state_value(in_place=False)
    v15_a, v13_b, v15_b = solve_exercise_4_2(values)
    print('\nExercise 4.2 Results:')
    print(f' (a) v_pi(15) = {v15_a:.6f}')
    print(f' (b) v_pi(13) = {v13_b:.6f}, v_pi(15) = {v15_b:.6f}')
