import argparse
import yaml
from easydict import EasyDict as edict
import os
import shutil


BASE_CONFIGS_PATH = r'./config/base.yaml'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, default=None, help='Mode Name')
    parser.add_argument('--mode', type=str, default='train', help='Mode Name')
    parser.add_argument('--cfg', type=str, default=None, help='Config Path')
    args = parser.parse_args()

    with open(BASE_CONFIGS_PATH, 'r') as cfgs_file:
        base_cfgs = yaml.load(cfgs_file, Loader=yaml.FullLoader)
    base_cfgs = edict(base_cfgs)

    if args.cfg is not None:
        with open(args.cfg, 'r') as cfgs_file:
            cfgs = yaml.load(cfgs_file, Loader=yaml.FullLoader)
    else:
        cfgs = {}
    cfgs = edict(cfgs)
    cfgs.update(base_cfgs)

    cfgs.method = args.method if args.method is not None else None
    cfgs.mode = args.mode if args.mode is not None else None


    if args.mode == 'train':
        os.makedirs(cfgs.results_path, exist_ok=True)
        shutil.copy(BASE_CONFIGS_PATH, f'{cfgs.results_path}.backup')
        shutil.copy(args.cfg, f'{cfgs.results_path}.backup')
        for subdir in cfgs.results_subdirs:
            os.makedirs(subdir, exist_ok=True)

        if args.method == 'DDPG':
            from scripts.train_ddpg import Trainer
        elif args.method == 'PPO':
            from scripts.train_ppo import Trainer
        trainer = Trainer(cfgs)
        trainer.training()
    elif args.mode == 'test':
        if args.method == 'DDPG':
            from scripts.test_ddpg import Tester
        elif args.method == 'PPO':
            from scripts.test_ppo import Tester
        tester = Tester(cfgs)
        tester.testing()
    elif args.mode == 'eval':
        if args.method == 'DDPG':
            from scripts.eval_ddpg import Evaluator
        elif args.method == 'PPO':
            from scripts.eval_ppo import Evaluator
        evaluator = Evaluator(cfgs)
        evaluator.evaluating()
    else:
        raise NotImplementedError
