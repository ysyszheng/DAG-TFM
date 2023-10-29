import sys
import random
import numpy as np
import torch
import os
from typing import List
import itertools


def fix_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def log(*args):
    COLOR_RED = "\033[91m"
    COLOR_RESET = "\033[0m"
    fn = sys._getframe().f_back.f_code.co_filename
    ln = sys._getframe().f_back.f_lineno
    highlight_msg = 'File \"%s\", line %d\n' % (fn, ln)

    colored_msg = COLOR_RED + highlight_msg + COLOR_RESET
    print(colored_msg, *args)


def normize(x, mean, std):
    return (x - mean) / std


def unormize(x, mean, std):
    return x * std + mean


def get_dist_from_margin(b, m, q):
    assert b >= 1 and m >= b and len(q) == m
    event_set = itertools.combinations(range(m), b)
    if b == 1:
        log(m,b)
        return event_set, q

    if m == b:
        log(m,b)
        return event_set, [1]
    
    if b > 1 and (b + 1) <= m and m < 2 * b:
        log(m,b)
        b_prime = m - b
        q_tilde = [1 - q[i] for i in range(len(q))]
        event_set_tilde, p_tilde = get_dist_from_margin(b_prime, m, q_tilde)
        log(m,b)
        p = [0] * len(list(event_set))
        for i, combo in enumerate(event_set):
            for j, combo_tilde in enumerate(event_set_tilde):
                if not set(combo).isdisjoint(set(combo_tilde)):
                    p[i] = p_tilde[j]
        return event_set, p

    if b > 1 and m >= 2 * b:
        log(m,b)
        sorted_indices = sorted(range(len(q)), key=lambda i: q[i], reverse=True)
        sorted_q = [q[i] for i in sorted_indices]

        q_tilde = [0 for _ in range(m-1)]
        for i in range(b-1):
            q_tilde[i] = (sorted_q[i]-sorted_q[m-1])/(1-sorted_q[m-1])
        for i in range(b-1, m-1):
            q_tilde[i] = (sorted_q[i])/(1-sorted_q[m-1])
        event_set_tilde, p_tilde = get_dist_from_margin(b, m-1, q_tilde)
        log(m,b)
        
        p = [0] * len(list(event_set))
        for i, combo in enumerate(event_set):
            if set(combo) == set(range(b-1)+[m-1]):
                p[i] = sorted_q[m-1]
            if (m-1) not in combo:
                for j, combo_tilde in enumerate(event_set_tilde):
                    if set(combo_tilde) == set(combo):
                        p[i] = (1-sorted_q[m-1])*p_tilde[j]

        for i, combo in enumerate(event_set):
            event_set[i] = tuple([sorted_indices[i] for i in combo])
        return event_set, p
    else:
        raise NotImplementedError


if __name__ == '__main__':
    log('hello world')
