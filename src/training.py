import sys
import torch
import numpy as np

from IPython.display import clear_output
from tqdm import tqdm

from utils import DEVICE, plot_history, metrics_bootstrap_smoothing
from prompt_samplers import PromptSampler


@torch.no_grad()
def linear_val(model, prompt_sampler: PromptSampler, criterion, **sampler_cfg):
    loss_fn = criterion(reduction="none")
    model.eval()
    xs_eval = prompt_sampler.sample_data(**sampler_cfg)
    xs_other = prompt_sampler.sample_data(**sampler_cfg)
    batch_size, prompt_len, dim = xs_eval.shape
    res = torch.zeros(batch_size, prompt_len // 2)
    for i in range(prompt_len // 2):
        xs_stack = torch.cat((xs_eval[:, :2 * i, :], xs_other[:, 2 * i:, :]), dim=1)
        prompt, ys = prompt_sampler.construct_prompt(xs=xs_stack, test=False)

        pred = model(prompt, inds=[i])
        res[:, i] = loss_fn(pred, ys)[:, i]

    model.train()
    return metrics_bootstrap_smoothing(res)


@torch.no_grad()
def mnist_val(model, prompt_sampler: PromptSampler, criterion, **sampler_cfg):
    loss_fn = criterion(reduction="none")
    model.eval()
    xs_eval, ys_eval = prompt_sampler.eval(**sampler_cfg)
    xs_other, ys_other = prompt_sampler.sample_data(**sampler_cfg)
    batch_size, prompt_len, dim = xs_eval.shape
    res = torch.zeros(batch_size, prompt_len)
    for i in range(prompt_len):
        xs_stack = torch.cat((xs_eval[:, :i, :], xs_other[:, i:, :]), dim=1)
        ys_stack = torch.cat((ys_eval[:, :i], ys_other[:, i:]), dim=1)
        prompt, ys = prompt_sampler.construct_prompt(xs=xs_stack, ys=ys_stack)

        pred = model(prompt, inds=[i]).squeeze()
        res[:, i] = loss_fn(pred, ys[:, i])
    model.train()
    return metrics_bootstrap_smoothing(res)


def train(model, optimizer, criterion, prompt_sampler, batch_size=64, steps=500_000, exp_name="linear"):
    loss_fn = criterion(reduction="mean")
    val_fn = linear_val if exp_name == "linear" else mnist_val
    loss_log = []
    model.train()
    tqdm_length = 2_000
    with tqdm(total=tqdm_length, file=sys.stdout) as prbar:
        for step in range(steps):
            prompt, target = prompt_sampler(device=DEVICE, batch_size=batch_size)

            optimizer.zero_grad()
            output = model(prompt).squeeze()
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()
            prompt_sampler.curriculum_update()

            loss_log.append(loss.item())
            prbar.update(1)

            if step > 0 and step % tqdm_length == 0:
                val_metrics = val_fn(model, prompt_sampler, criterion)
                clear_output()
                torch.save(model, f"./models/{model.name}_{step}.pt")
                plot_history(loss_log, val_metrics, title='loss')
                print("Steps passed: {0}, loss: {1}".format(step, np.mean(loss_log)))
                prbar.reset()
    return loss_log
