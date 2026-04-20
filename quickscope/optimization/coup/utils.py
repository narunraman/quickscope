import os
import math
import pickle
import numpy as np


colors = ['#377eb8', '#ff7f00', '#4daf4a', '#f781bf', '#a65628', '#984ea3', '#999999', '#e41a1c', '#dede00']

color_schemes = [['#377eb8', '#629fd0', '#9dc3e2'], 
                ['#ff7f00', '#ffa64d', '#ffcc99'], 
                ['#4daf4a', '#72c36f', '#a7d9a5'],
                ['#f781bf', '#f99fcf', '#fccfe7'],
                ['#a65628', '#d27a46', '#e1a684'],
                ['#984ea3', '#bd87c5', '#d9bade']]
                # ['#999999', '#bfbfbf', '#d9d9d9']]

fs = {'axis': 22,
      'title': 25,
      'ticks': 20,
      'legend': 15}

lw = {'main': 5,
      'small': 2,
      'tiny': .5,
      'fat': 8}


day_in_s = 60 * 60 * 24


################# utility functions #################

def u_satcomp(t, T, c):
    if t < T:
        return 1 - t / T / c
    else:
        return 0

def u_unif(t, k0): # Uniform
    if t < k0:
        return 1 - t / k0
    else:
        return 0

def u_exp(t, k0): # Exponential 
    return math.exp(- t / k0)

def u_pareto(t, k0, a): # Pareto 
    if t < k0:
        return 1
    else:
        return (k0 / t) ** a

def u_ll(t, k0, a): # Log-Laplace
    if  t < k0:
        return 1 - (t / k0) ** a / 2
    else:
        return (k0 / t) ** a / 2

def u_gll(t, k0, a, b): # generalized log-Laplace
    if  t < k0:
        return 1 - a * (t / k0) ** b / (a + b)
    else:
        return b * (k0 / t) ** a / (a + b)

def u_poly(t, k0, a):
    if t < k0:
        return 1 - (t / k0)**a
    else:
        return 0

def u_forward(t, k0, k1, a):
    if t <= k0:
        return 1
    if k0 < t and t < k1:
        return ((k1 - t) / (k1 - k0)) ** a
    else:
        return 0

def u_backward(t, k0, k1, a):
    if t <= k0:
        return 1
    if k0 < t and t < k1:
        return 1 - ((t - k0) / (k1 - k0)) ** (1/a)
    else:
        return 0

def u_step(t, k0):
    if t < k0:
        return 1
    else:
        return 0


def u_lin(t, k0, k1):
    if t < k0: 
        return 1
    elif t < k1:
        return (k1 - t) / (k1 - k0)
    else:
        return 0


def u_geometric(t, k0, k1):
    if t < k0:
        return 1
    elif t < k1:
        return math.log(k1 / t) / math.log(k1 / k0)
    else:
        return 0


utility_function_by_name = {
    'u_satcomp': u_satcomp,
    'u_unif': u_unif,
    'u_exp': u_exp,
    'u_pareto': u_pareto,
    'u_ll': u_ll,
    'u_gll': u_gll,
    'u_poly': u_poly,
    'u_forward': u_forward,
    'u_backward': u_backward,
    'u_step': u_step,
    'u_lin': u_lin,
    'u_geometric': u_geometric,
}


def u_to_str(utility_function):
    """ return a nice string representation of a utility function and its parameters """
    u_fn, u_params = utility_function
    if type(u_params) is dict:
        param_str = ",".join("{}={}".format(k, v) for k, v in u_params.items())
    else:
        param_str = ",".join("{}".format(v) for v in u_params)
    return "{}(".format(u_fn.__name__) + param_str + ")"


def parse_u(s):
    """ for parsing utility functions from command line arguments """
    ss = s.split(" ")
    u_fn = utility_function_by_name[ss[0]]
    u_params = {}
    for p in ss[1:]:
        pname, pval = p.split("=")
        u_params[pname] = float(pval)
    
    expected_params = [p for p in u_fn.__code__.co_varnames if p != 't']
    for p in expected_params:
        if p not in u_params:
            print(f"utility function {ss[0]} requires parameters {expected_params} but is missing {p}.")
            exit()

    return (u_fn, u_params)


##################################


def ecdf(data):
    """ empirical CDF """
    x = np.sort(data)
    y = np.arange(len(x)) / float(len(x))
    return x, y


def choose_max(main_array, secondary_array):
    """ Get the maximum of main_array, breaking ties according to secondary_array """
    choice_array = np.full(main_array.shape, -np.inf)
    choice_array[main_array == main_array.max()] = secondary_array[main_array == main_array.max()]
    return np.random.choice(np.flatnonzero(choice_array == choice_array.max()))


def random_seed(n=None, imax=1e8):
    return np.random.randint(low=0, high=imax, size=n)


def ensure_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def safe_save(data, path):
    with open(path + '.tmp', 'wb') as f: # nonatomic write to temp file
        pickle.dump(data, f)
    os.replace(path + '.tmp', path) # then atomic rename if successful


