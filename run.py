import argparse
import yaml
from easydict import EasyDict as edict
import os


BASE_CONFIGS_PATH = r'./config/base.yaml'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, default=None, help='Mode Name')
    parser.add_argument('--mode', type=str, default='train', help='Mode Name')
    parser.add_argument('--cfg', type=str, default=None, help='Config Path')
    parser.add_argument('--file_name', type=str,
                        default=None, help='File Name')
    args = parser.parse_args()

    with open(BASE_CONFIGS_PATH, 'r') as cfg_file:
        base_cfgs = yaml.load(cfg_file, Loader=yaml.FullLoader)
    base_cfgs = edict(base_cfgs)

    if args.cfg is not None:
        with open(args.cfg, 'r') as cfg_file:
            cfgs = yaml.load(cfg_file, Loader=yaml.FullLoader)
    else:
        cfgs = {}
    cfgs = edict(cfgs)
    cfgs.update(base_cfgs)

    cfgs.mode = args.mode if args.mode is not None else None
    cfgs.fn = args.file_name if args.file_name is not None else None

    if args.mode == 'train':
        if args.method == 'DDPG':
            from scripts.train_ddpg import Trainer
        elif args.method == 'PPO':
            from scripts.train_ppo import Trainer
        trainer = Trainer(cfgs)
        trainer.training()
    if args.mode == 'test':
        if args.method == 'DDPG':
            from scripts.test_ddpg import Tester
        elif args.method == 'PPO':
            from scripts.test_ppo import Tester
        tester = Tester(cfgs)
        tester.testing()
    else:
        raise NotImplementedError
