from abc import ABC, abstractmethod

import torch
import numpy as np
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader

from src.utils import load_mnist


class PromptSampler(ABC):
    def __init__(self, dim=20, prompt_len=20, **curriculum_cfg):
        self.d_cur = curriculum_cfg.get("dim_start", dim)
        self.d_max = curriculum_cfg.get("dim_max", dim)
        self.len_cur = curriculum_cfg.get("len_start", prompt_len)
        self.len_max = curriculum_cfg.get("len_max", prompt_len)
        self.d_add = curriculum_cfg.get("dim_add", 1)
        self.len_add = curriculum_cfg.get("len_add", 1)
        self.update_curriculum_steps = curriculum_cfg.get("every_steps", 2_000)
        self.curriculum = curriculum_cfg.get("curriculum", False)
        self.steps_done = 0

    def curriculum_update(self):
        if not self.curriculum:
            return
        self.steps_done += 1
        if self.steps_done % self.update_curriculum_steps == 0:
            self.d_cur = min(self.d_cur + self.d_add, self.d_max)
            self.len_cur = min(self.len_cur + self.len_add, self.len_max)

    @abstractmethod
    def sample_data(self, batch_size=64, device="cuda"):
        pass

    @abstractmethod
    def eval(self, xs, test=False):
        pass

    @abstractmethod
    def construct_prompt(self, batch_size=64, device="cuda", xs=None, ys=None, test=False):
        pass

    def __call__(self, *args, **kwargs):
        return self.construct_prompt(*args, **kwargs)


class LinearPromptSampler(PromptSampler):
    def sample_data(self, batch_size=64, device="cuda"):
        xs = torch.randn((batch_size, self.len_cur, self.d_max), device=device)
        if self.d_cur < self.d_max:
            xs[:, :, self.d_cur:] = 0
        return xs

    def eval(self, xs, test=False):
        batch_size, prompt_len, dim = xs.shape
        w = torch.randn((batch_size, dim, 1), device=xs.device)
        if test:
            w = torch.normal(10, 10, (batch_size, dim, 1), device=xs.device)
        return (xs @ w)[:, :, 0]

    def construct_prompt(self, batch_size=64, device="cuda", xs=None, ys=None, test=False):
        if xs is None:
            xs = self.sample_data(batch_size, device)
        if ys is None:
            ys = self.eval(xs, test)
        batch_size, prompt_len, dim = xs.shape
        ys_wide = torch.cat(
            (
                ys.view(batch_size, prompt_len, 1),
                torch.zeros((batch_size, prompt_len, dim - 1), device=device),
            ),
            dim=2,
        )
        prompt = torch.stack((xs, ys_wide), dim=2)
        prompt = prompt.view(batch_size, 2 * prompt_len, dim)
        return prompt, ys


class LinearPromptDualSampler(PromptSampler):
    def sample_data(self, batch_size=64, device="cuda"):
        xs = torch.randn((batch_size, 2 * self.len_cur, self.d_max), device=device)
        if self.d_cur < self.d_max:
            xs[:, :, self.d_cur:] = 0
        return xs

    def eval(self, xs, test=False):
        batch_size, prompt_len, dim = xs.shape
        w = torch.randn((batch_size, dim, 1), device=xs.device)
        sums = xs[:, ::2, :] + xs[:, 1::2, :]
        if test:
            w = torch.normal(10, 10, (batch_size, dim, 1), device=xs.device)
        return (sums @ w)[:, :, 0]

    def construct_prompt(self, batch_size=64, device="cuda", xs=None, ys=None, test=False):
        if xs is None:
            xs = self.sample_data(batch_size, device)
        if ys is None:
            ys = self.eval(xs, test)
        batch_size, prompt_len, dim = xs.shape
        ys_wide = torch.cat(
            (
                ys.view(batch_size, self.len_cur, 1),
                torch.zeros((batch_size, self.len_cur, dim - 1), device=device),
            ),
            dim=2,
        )
        prompt = torch.zeros(batch_size, prompt_len + self.len_cur, dim, device=device)
        prompt[:, ::3, :] = xs[:, ::2, :]
        prompt[:, 1::3, :] = xs[:, 1::2, :]
        prompt[:, 2::3, :] = ys_wide
        return prompt, ys


class MnistPromptSampler(PromptSampler):
    class Mnist(torch.utils.data.Dataset):
        def __init__(self, X, y):
            self.X, self.y = X, y

        def __len__(self):
            return self.X.shape[0]

        def __getitem__(self, idx):
            return self.X[idx], self.y[idx]

    def __init__(self, dim=20, prompt_len=20, **curriculum_cfg):
        super().__init__(dim, prompt_len, **curriculum_cfg)
        X_train, y_train, X_val, y_val, X_test, y_test = load_mnist(flatten=True)
        self.X_train = np.concatenate([X_train, X_val], axis=0)
        self.X_train = torch.tensor(X_train).float()
        self.y_train = np.concatenate([y_train, y_val], axis=0)
        self.y_train = torch.tensor(y_train).float()
        self.X_test = torch.tensor(X_test).float()
        self.y_test = torch.tensor(y_test).float()
        self.train_idx = np.arange(0, y_train.shape[0])
        self.test_idx = np.arange(0, y_test.shape[0])

    def sample_data(self, batch_size=64, device="cuda"):
        self.device = device
        self.batch_size = batch_size
        possible_classes = np.arange(0, self.d_cur)
        batch_X = torch.empty((0, self.len_cur, 28 * 28), device=device)
        batch_y = torch.empty((0, self.len_cur), device=device)
        for i in range(batch_size):
            idx = np.random.choice(self.train_idx[np.isin(self.y_train, possible_classes)], size=self.len_cur)
            # print(batch_y.shape, self.y_train[idx].view(1, self.len_cur, -1).shape)
            batch_X = torch.vstack([batch_X, self.X_train[idx].to(device).view(1, self.len_cur, -1)])
            batch_y = torch.vstack([batch_y, self.y_train[idx].to(device).view(1, self.len_cur)])
        return batch_X, batch_y

    def eval(self, xs=None, test=False):
        possible_classes = np.arange(0, self.d_cur)
        batch_X = torch.empty((0, self.len_cur, 28 * 28), device=self.device)
        batch_y = torch.empty((0, self.len_cur), device=self.device)
        for i in range(self.batch_size):
            idx = np.random.choice(self.test_idx[np.isin(self.y_test, possible_classes)], size=self.len_cur)
            batch_X = torch.cat([batch_X, self.X_test[idx].to(self.device).view(1, self.len_cur, -1)])
            batch_y = torch.cat([batch_y, self.y_test[idx].to(self.device).view(1, self.len_cur)])
        return batch_X, batch_y

    def construct_prompt(self, batch_size=64, device="cuda", xs=None, ys=None):
        if xs is None:
            xs, ys = self.sample_data()
        batch_size, prompt_len, dim = xs.shape
        ys_wide = torch.cat(
            (
                ys.view(batch_size, prompt_len, 1),
                torch.zeros((batch_size, prompt_len, dim - 1), device=device),
            ),
            dim=2,
        )
        prompt = torch.stack((xs, ys_wide), dim=2)
        prompt = prompt.view(batch_size, 2 * prompt_len, dim)
        return prompt, ys
