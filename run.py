from config.cfg import cfg
from utils.log import log
import os
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type = str, default = 'DDPG', help = 'Mode Name')
    parser.add_argument('--mode', type = str, default = 'train', help = 'Mode Name')
    parser.add_argument('--fn', type = str, default = None, help = 'File Name')
    args = parser.parse_args()

    if args.mode is not None:
        cfg['mode'] = args.mode
    if args.fn is not None:
        cfg['fn'] = args.fn

    dirs = ['./img', './models', './data', './rewards']
    for dir in dirs:
        if not os.path.exists(dir):
            os.makedirs(dir)
    
    if args.model == 'DDPG':
        if args.mode == 'train':
            from scripts.train import Trainer
            trainer = Trainer(cfg)
            trainer.training()
        elif args.mode == 'test' or args.mode == 'verify' or args.mode == 'optim':
            from scripts.test import Tester
            tester = Tester(cfg)
            tester.testing()
        else:
            raise NotImplementedError
    elif args.model == 'Linear':
        from scripts.linear import Linear
        linear = Linear(cfg)
        linear.optim()
    else:
        raise NotImplementedError
