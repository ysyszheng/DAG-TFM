import argparse
import yaml
from easydict import EasyDict as edict
from utils.cfgs import handle_cfgs


# global configs
BASE_CONFIGS_PATH = r'./config/base.yaml'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, required=True, help='Mode Name')
    parser.add_argument('--mode', type=str, required=True, default='train', help='Mode Name')
    parser.add_argument('--cfg', type=str, required=True, help='Config Path')
    parser.add_argument('--seed', type=int)
    parser.add_argument('--max_agents_num', type=int)
    parser.add_argument('--lambd', type=float)
    parser.add_argument('--delta', type=float)
    parser.add_argument('--burn_flag', type=str)
    parser.add_argument('--b', type=int)
    parser.add_argument('--a', type=float)
    parser.add_argument('--clip_value', type=float)
    parser.add_argument('--norm_value', type=float)
    args = parser.parse_args()
    args = argparse.Namespace(**{k: v for k, v in vars(args).items() if v is not None})

    with open(BASE_CONFIGS_PATH, 'r') as cfgs_file:
        base_cfgs = yaml.load(cfgs_file, Loader=yaml.FullLoader)
    base_cfgs = edict(base_cfgs)

    if args.cfg is not None:
        with open(args.cfg, 'r') as cfgs_file:
            cfgs = yaml.load(cfgs_file, Loader=yaml.FullLoader)
    else:
        cfgs = {}
        raise ValueError('Param `cfgs` is None.')

    cfgs = edict(cfgs)
    cfgs.update(base_cfgs)
    cfgs = handle_cfgs(cfgs, vars(args))


    if cfgs.mode == 'train':
        if cfgs.method == 'DDPG':
            from scripts.train_ddpg import Trainer
        elif cfgs.method == 'PPO':
            from scripts.train_ppo import Trainer
        elif cfgs.method == 'NN':
            from scripts.train_nn import Trainer
        elif cfgs.method == 'DNN':
            from scripts.train_dnn import Trainer
        elif cfgs.method == 'ES':
            from scripts.train_es import Trainer
        elif cfgs.method == 'CMAES':
            from scripts.train_cmaes import Trainer
        else:
            raise NotImplementedError(f'Method {cfgs.method} is not implemented in {cfgs.mode} mode.')

        trainer = Trainer(cfgs)
        trainer.training()

    elif cfgs.mode == 'test':
        if cfgs.method == 'DDPG':
            from scripts.test_ddpg import Tester
        elif cfgs.method == 'PPO':
            from scripts.test_ppo import Tester
        elif cfgs.method == 'NN':
            from scripts.test_nn import Tester
        elif cfgs.method == 'ES':
            from scripts.test_es import Tester
        else:
            raise NotImplementedError(f'Method {cfgs.method} is not implemented in {cfgs.mode} mode.')
        
        tester = Tester(cfgs)
        tester.testing()

    elif cfgs.mode == 'eval':
        if cfgs.method == 'DDPG':
            from scripts.eval_ddpg import Evaluator
        elif cfgs.method == 'PPO':
            from scripts.eval_ppo import Evaluator
        elif cfgs.method == 'NN':
            from scripts.eval_nn import Evaluator
        elif cfgs.method == 'ES':
            from scripts.eval_es import Evaluator
        elif cfgs.method == 'CMAES':
            from scripts.eval_cmaes import Evaluator
        else:
            raise NotImplementedError(f'Method {cfgs.method} is not implemented in {cfgs.mode} mode.')

        evaluator = Evaluator(cfgs)
        evaluator.evaluating()

    elif cfgs.mode == 'show':
        if cfgs.method == 'ES':
            from scripts.train_es import Trainer
            Trainer(cfgs).plot_strategy()
        else:
            raise NotImplementedError(f'Method {cfgs.method} is not implemented in {cfgs.mode} mode.')

    else:
        raise NotImplementedError(f'Mode {cfgs.mode} is not implemented.')
