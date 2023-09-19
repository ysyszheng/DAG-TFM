import sys
import random
import numpy as np
import torch
import os


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


if __name__ == '__main__':
    log('hello world')
