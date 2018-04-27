import numpy as np
from ._data_generator import make_rectangular
from ._data_generator import make_triangular

def get_decision_tree_data_1(n_samples=1000):
    return _get_decision_tree_data(
        decision_tree_data_1_profile, n_samples)

def get_decision_tree_data_2(n_samples=1000):
    return _get_decision_tree_data(
        decision_tree_data_2_profile, n_samples)

# (type, n_samples, label, x_b, x_e, y_b, y_e)
decision_tree_data_1_profile = [
    ('rec', 25, 1, 0, 5, 0, 5),
    ('rec', 15, 0, 5, 10, 0, 3),
    ('rec', 3, 0, 5, 7, 3, 4.5),
    ('rec', 9.5, 1, 7, 10, 3, 6.5),
    ('rec', 1, 1, 5, 7, 4.5, 5),    
    ('rec', 10, 1, 2, 7, 5, 7),    
    ('rec', 4, 0, 0, 2, 5, 7),    
    ('rec', 13.5, 0, 0, 4.5, 7, 10),
    ('rec', 7.5, 1, 4.5, 7, 7, 10),    
    ('rec', 9.5, 0, 7, 10, 6.5, 10),
]

decision_tree_data_2_profile = [
    ('rec', 32, 1, 0, 7, 0, 4),
    ('rec', 9.5, 0, 7, 10, 0, 3.5),
    ('rec', 4.5, 1, 7, 10, 3.5, 5),
    ('rec', 0.5, 1, 7.5, 8, 4, 5),
    ('rec', 7.5, 0, 7.5, 10, 5, 8),
    ('rec', 4.5, 1, 8, 10, 8, 10),
    ('rec', 1, 0, 7.5, 8, 8, 10),
    ('rec', 16.5, 0, 2, 7.5, 7, 10),
    ('rec', 12, 0, 0, 2, 4, 10),
    ('upper', 8.25, 1, 2, 7.5, 4, 7),
    ('lower', 8.25, 0, 2, 7.5, 4, 7),
]

def _get_decision_tree_data(profile, n_samples=1000):
    # set num of samples
    factor = n_samples / sum((p[1] for p in profile))

    X_array = []
    color_array = []

    for i, p in enumerate(profile):

        n_samples_ = int(p[1] * factor)
        #label = i
        label = p[2]

        if p[0] == 'rec':
            X, color = make_rectangular(n_samples_,
                label=label, x_b=p[3], x_e=p[4], y_b=p[5], y_e=p[6])
        elif p[0] == 'upper':
            X, color = make_triangular(n_samples_, True,
                label=label, x_b=p[3], x_e=p[4], y_b=p[5], y_e=p[6])
        elif p[0] == 'lower':
            X, color = make_triangular(n_samples_, False,
                label=label, x_b=p[3], x_e=p[4], y_b=p[5], y_e=p[6])
        else:
            raise ValueError('Profile type error. Type={}'.format(p[0]))

        X_array.append(X)
        color_array.append(color)

    X = np.concatenate(X_array)
    color = np.concatenate(color_array)

    return X, color