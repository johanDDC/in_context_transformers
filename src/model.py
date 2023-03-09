from transformers import GPT2Model, GPT2Config
import torch.nn as nn
import torch

class LinearLearnerModel(nn.Module):
    def __init__(self, prompt_dim, n_positions, embedding_dim=256, n_layer=12, n_head=4):
        super().__init__()
        configuration = GPT2Config(
            n_positions=2 * n_positions,
            n_embd=embedding_dim,
            n_layer=n_layer,
            n_head=n_head,
            resid_pdrop=0.0,
            embd_pdrop=0.0,
            attn_pdrop=0.0,
            use_cache=False,
        )
        self.name = "linear_learner"
        self.n_positions = n_positions
        self.n_dims = prompt_dim
        self.in_embeddings = nn.Linear(prompt_dim, embedding_dim)
        self.gpt = GPT2Model(configuration)
        self.out_embeddings = nn.Linear(embedding_dim, 1)

    def forward(self, prompt, inds=None):
        inds = torch.arange(prompt.shape[1] // 2) if inds is None else torch.tensor(inds)
        mapped_prompt = self.in_embeddings(prompt)
        output = self.gpt(inputs_embeds=mapped_prompt).last_hidden_state
        pred = self.out_embeddings(output)
        return pred[:, ::2, 0][:, inds]


class LinearDualLearnerModel(nn.Module):
    def __init__(self, prompt_dim, n_positions, embedding_dim=256, n_layer=12, n_head=4):
        super().__init__()
        configuration = GPT2Config(
            n_positions=3 * n_positions,
            n_embd=embedding_dim,
            n_layer=n_layer,
            n_head=n_head,
            resid_pdrop=0.0,
            embd_pdrop=0.0,
            attn_pdrop=0.0,
            use_cache=False,
        )
        self.name = "linear_dual_learner"
        self.n_positions = n_positions
        self.n_dims = prompt_dim
        self.in_embeddings = nn.Linear(prompt_dim, embedding_dim)
        self.gpt = GPT2Model(configuration)
        self.out_embeddings = nn.Linear(embedding_dim, 1)

    def forward(self, prompt, inds=None):
        inds = torch.arange(prompt.shape[1] // 3) if inds is None else torch.tensor(inds)
        mapped_prompt = self.in_embeddings(prompt)
        output = self.gpt(inputs_embeds=mapped_prompt).last_hidden_state
        pred = self.out_embeddings(output)
        return pred[:, ::3, 0][:, inds]


class MnistMSEClassifierModel(nn.Module):
    def __init__(self, n_positions, prompt_dim=28*28, embedding_dim=256, n_layer=12, n_head=4):
        super().__init__()
        configuration = GPT2Config(
            n_positions=2 * n_positions,
            n_embd=embedding_dim,
            n_layer=n_layer,
            n_head=n_head,
            resid_pdrop=0.0,
            embd_pdrop=0.0,
            attn_pdrop=0.0,
            use_cache=False,
        )
        self.name = "mnist_mse"
        self.n_positions = n_positions
        self.n_dims = prompt_dim
        self.in_embeddings = nn.Linear(prompt_dim, embedding_dim)
        self.gpt = GPT2Model(configuration)
        self.out_embeddings = nn.Linear(embedding_dim, 1)

    def forward(self, prompt, inds=None):
        inds = torch.arange(prompt.shape[1] // 2) if inds is None else torch.tensor(inds)
        mapped_prompt = self.in_embeddings(prompt)
        output = self.gpt(inputs_embeds=mapped_prompt).last_hidden_state
        pred = self.out_embeddings(output)
        return pred[:, ::2, 0][:, inds]


class MnistClassifierModel(nn.Module):
    def __init__(self, n_positions, prompt_dim=28*28, embedding_dim=256, n_layer=12, n_head=4):
        super().__init__()
        configuration = GPT2Config(
            n_positions=2 * n_positions,
            n_embd=embedding_dim,
            n_layer=n_layer,
            n_head=n_head,
            resid_pdrop=0.0,
            embd_pdrop=0.0,
            attn_pdrop=0.0,
            use_cache=False,
        )
        self.name = "mnist_class"
        self.n_positions = n_positions
        self.n_dims = prompt_dim
        self.in_embeddings = nn.Linear(prompt_dim, embedding_dim)
        self.gpt = GPT2Model(configuration)
        self.out_embeddings = nn.Linear(embedding_dim, 10)
        self.softmax = nn.Softmax()

    def forward(self, prompt, inds=None):
        inds = torch.arange(prompt.shape[1] // 2) if inds is None else torch.tensor(inds)
        mapped_prompt = self.in_embeddings(prompt)
        output = self.gpt(inputs_embeds=mapped_prompt).last_hidden_state
        pred = self.out_embeddings(output)
        return self.softmax(pred[:, ::2, 0][:, inds])