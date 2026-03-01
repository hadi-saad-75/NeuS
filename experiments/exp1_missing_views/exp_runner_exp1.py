import os
import argparse
import torch
from pyhocon import ConfigFactory

from exp_runner import Runner as BaseRunner
from experiments.exp1_missing_views.dataset_subsample import SubsampleDataset


class RunnerExp1(BaseRunner):
    def __init__(self, conf_path, mode='train', case='', view_ratio=1.0, is_continue=False):
        self.view_ratio = view_ratio
        super().__init__(conf_path, mode, case, is_continue)

    def _init_dataset(self):
        self.dataset = SubsampleDataset(self.conf['dataset'], self.view_ratio)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', type=str, required=True)
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--case', type=str, required=True)
    parser.add_argument('--view_ratio', type=float, default=1.0)
    parser.add_argument('--is_continue', action="store_true")
    parser.add_argument('--gpu', type=int, default=0)

    args = parser.parse_args()

    torch.cuda.set_device(args.gpu)

    runner = RunnerExp1(
        conf_path=args.conf,
        mode=args.mode,
        case=args.case,
        view_ratio=args.view_ratio,
        is_continue=args.is_continue
    )

    if args.mode == 'train':
        runner.train()
    elif args.mode == 'validate_mesh':
        runner.validate_mesh(world_space=True, resolution=512)


if __name__ == '__main__':
    main()
