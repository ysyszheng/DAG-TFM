from config.cfg import cfg
from utils.log import log
import os
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type = str, default = None, help = 'Mode Name')
    parser.add_argument('--fn', type = str, default = None, help = 'File Name')
    args = parser.parse_args()

    if args.mode is not None:
        cfg['mode'] = args.mode
    if args.fn is not None:
        cfg['fn'] = args.fn

    dirs = ['./img', './models', './data']
    for dir in dirs:
        if not os.path.exists(dir):
            os.makedirs(dir)

    if args.mode == 'train':
        from scripts.train import Trainer
        trainer = Trainer(cfg)
        trainer.training()
    elif args.mode == 'test':
        from scripts.test import Tester
        tester = Tester(cfg)
        tester.testing()
    elif args.mode == 'verify':
        from scripts.verify import Verifier
        verifier = Verifier(cfg)
        verifier.verifying()
    else:
        raise NotImplementedError
