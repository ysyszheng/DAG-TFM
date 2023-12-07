from easydict import EasyDict as edict
import os


def generate_path(cfgs):
    os.makedirs(cfgs.path.results_path, exist_ok=True)
    os.makedirs(cfgs.path.model_path, exist_ok=True)
    os.makedirs(cfgs.path.img_path, exist_ok=True)
    os.makedirs(cfgs.path.data_path, exist_ok=True)
    os.makedirs(cfgs.path.log_path, exist_ok=True)


def recursive_update(a, b):
    '''merge cfgs `b` to cfgs `a`'''
    for key in b.keys():
        if key in a.keys():
            dict_a = type(a[key]) is edict or type(a[key]) is dict
            dict_b = type(b[key]) is edict or type(b[key]) is dict
            if dict_a and dict_b:
                a[key] = recursive_update(a[key], b[key])
            elif not dict_a and not dict_b:
                a[key] = b[key]
            else:
                raise AttributeError(f'Type mismatch for key: {key}')
        else:
            a[key] = b[key]
    return a


def handle_cfgs(default, custom, verbose=True):
    '''merge cfgs and generate path'''
    # update default cfgs using custom cfgs
    cfgs = recursive_update(edict(default.copy()), edict(custom.copy()))

    # create path
    generate_path(cfgs)
    cfgs.path.cfg_path = cfgs.cfg
    del cfgs['cfg']

    if cfgs.burn_flag in ['non', 'log', 'poly']:
        if cfgs.burn_flag == 'non':
            cfgs.a = None
        cfgs.path.model_path = os.path.join(
            cfgs.path.model_path, 
            f'{cfgs.method}_{cfgs.lambd}_{cfgs.burn_flag}_{cfgs.a}.pth'
        )
        cfgs.path.img_path = os.path.join(
            cfgs.path.img_path , 
            f'{cfgs.method}_{cfgs.lambd}_{cfgs.burn_flag}_{cfgs.a}.png'
        )
        cfgs.path.log_path = os.path.join(
            cfgs.path.log_path, 
            f'{cfgs.method}_{cfgs.mode}_{cfgs.lambd}_{cfgs.burn_flag}_{cfgs.a}.log'
        )
    else:
        raise ValueError('Param `burn_flag` not in ["non", "log", "poly"].')

    if verbose:
        from utils.utils import init_logger
        logger = init_logger(__name__, cfgs.path.log_path, clear_file=True)
        logger.info('========== print cfgs ==========')
        logger.info(cfgs)
        logger.info('========== print cfgs ==========\n')

    return cfgs
