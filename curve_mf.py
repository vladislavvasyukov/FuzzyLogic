import numpy as np
from skfuzzy import trimf


def kinked_curve_mf(universe, curve_params_set):
    Y = np.zeros(len(universe))

    idx = np.nonzero(universe <= curve_params_set[0][0])[0]
    Y[idx] = curve_params_set[0][1]

    left = curve_params_set[0][0]
    y1 = curve_params_set[0][1]

    for i, (right, y2) in enumerate(curve_params_set):
        idx = np.nonzero(np.logical_and(universe >= left, universe <= right))[0]
        
        if y2 == y1:
            Y[idx] = y1
        else:
            left_ind = idx[0]
            right_ind = idx[-1]

            mid, up = (right_ind, y1) if y2 - y1 > 0 else (left_ind, y2)
            tr_val = trimf(idx, [left_ind, mid, right_ind])
            values = tr_val * abs(y2-y1) + up
            Y[idx] = values

        left, y1 = right, y2

    return Y
