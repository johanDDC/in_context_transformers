import argparse
import torch
import torch.nn as nn

import warnings

warnings.filterwarnings("ignore")

from model import LinearLearnerModel, MnistMSEClassifierModel, LinearDualLearnerModel
from training import train
from utils import DEVICE
from prompt_samplers import MnistPromptSampler, LinearPromptSampler, LinearPromptDualSampler


def linear_experiment(model, opt, **curriculum_settings):
    criterion = nn.MSELoss
    sampler = LinearPromptSampler(**curriculum_settings)
    return train(model, opt, criterion, sampler, steps=500_000, exp_name="linear")


def linear_dual_experiment(model, opt, **curriculum_settings):
    criterion = nn.MSELoss
    sampler = LinearPromptDualSampler(**curriculum_settings)
    return train(model, opt, criterion, sampler, steps=500_000, exp_name="linear")


def mnist_experiment(model, opt, **curriculum_settings):
    criterion = nn.MSELoss
    # sampler = LinearPromptSampler(**curriculum_settings)
    sampler = MnistPromptSampler(**curriculum_settings)
    return train(model, opt, criterion, sampler, steps=500_000, exp_name="mnist")


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("-e", "--exp", required=False, type=str, default="dual")
    args.add_argument("-c", "--curriculum", required=False, type=bool, default=True)
    args.add_argument("-ds", "--dim_start", required=False, type=int, default=5)
    args.add_argument("-dm", "--dim_max", required=False, type=int, default=20)
    args.add_argument("-ls", "--len_start", required=False, type=int, default=11)
    args.add_argument("-lm", "--len_max", required=False, type=int, default=20)
    args.add_argument("-us", "--update_steps", required=False, type=int, default=2_000)
    args = args.parse_args()

    if args.exp == "linear":
        model = LinearLearnerModel(20, 20).to(DEVICE)
        experiment = linear_experiment
    elif args.exp == "dual":
        model = LinearDualLearnerModel(20, 20).to(DEVICE)
        experiment = linear_dual_experiment
    else:
        model = MnistMSEClassifierModel(32).to(DEVICE)
        experiment = lambda **kwargs: mnist_experiment(**kwargs, prompt_len=16)
    opt = torch.optim.Adam(model.parameters(), 1e-4)

    experiment(model=model,
               opt=opt, curriculum=args.curriculum,
               dim_start=args.dim_start, dim_max=args.dim_max,
               len_start=args.len_start, len_max=args.len_max,
               update_curriculum_steps=args.update_steps)
    # # model = MnistMSEClassifierModel(32).to(DEVICE)
    #
    # opt = torch.optim.Adam(model.parameters(), 1e-4)
    #
    # # linear_experiment(model, opt, curriculum=True, dim_start=5, dim_max=20, len_start=11, len_max=20,
    # #                   every_steps=2_000)
    # linear_dual_experiment(model, opt, curriculum=True, dim_start=5, dim_max=20, len_start=11, len_max=20,
    #                   every_steps=3_000)
    # # mnist_experiment(model, opt, criterion, prompt_len=16, curriculum=True, dim_start=1, dim_max=10, len_start=5,
    # #                  len_max=20,
    # #                  every_steps=1_000)
