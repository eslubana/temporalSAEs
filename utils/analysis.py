import torch
from torch.utils.data import DataLoader
from model import GPT
from dgp import get_dataloader
from sae import SAEData, SAE
import pickle as pkl
import os
import json
import argparse

def get_model(path):
    state_dict = torch.load(os.path.join(path, 'latest_ckpt.pt'), map_location='cpu')
    cfg = state_dict['config']

    with open(os.path.join(path, 'grammar/PCFG.pkl'), 'rb') as f:
        pcfg = pkl.load(f)
    model = GPT(cfg.model, pcfg.vocab_size)
    model.load_state_dict(state_dict['net'])
    model.eval()
    dataloader = get_dataloader(
            language=cfg.data.language,
            config=cfg.data.config,
            alpha=cfg.data.alpha,
            prior_type=cfg.data.prior_type,
            num_iters=cfg.data.num_iters * cfg.data.batch_size,
            max_sample_length=cfg.data.max_sample_length,
            seed=cfg.seed,
            batch_size=cfg.data.batch_size,
            num_workers=0,
        )
    return pcfg, model, dataloader

def get_sae_data(path, idx, layer):
    cfg = get_config(path, idx)
    data = SAEData(model_dir=path, ckpt='latest_ckpt.pt', layer_name=layer, config=cfg['config'], device='cpu')
    dl = DataLoader(data, batch_size=10, shuffle=False, collate_fn=data.collate_fn)
    return dl

def get_module(model, layer):
    match layer:
        case "wte":   module = model.transformer.wte
        case "wpe":   module = model.transformer.wpe
        case "attn0": module = model.transformer.h[0].attn
        case "mlp0":  module = model.transformer.h[0].mlp
        case "res0":  module = model.transformer.h[0]
        case "attn1": module = model.transformer.h[1].attn
        case "mlp1":  module = model.transformer.h[1].mlp
        case "res1":  module = model.transformer.h[1]
        case "ln_f":  module = model.transformer.ln_f
    return module

def get_sae(path, idx):
    state_dict = torch.load(os.path.join(path, 'latest_ckpt.pt'), map_location='cpu')
    cfg = state_dict['config']
    embedding_size = cfg.model.n_embd
    args = json.load(open(os.path.join(path, f'sae_{idx}/config.json')))
    sae = SAE(embedding_size, args['exp_factor'] * embedding_size, pre_bias=args['pre_bias'], k=args['k'], sparsemax=args['sparsemax'], norm=args['norm']).to('cpu')
    state_dict = torch.load(os.path.join(path, f'sae_{idx}/model.pth'), map_location='cpu')
    sae.load_state_dict(state_dict)
    sae.eval()
    return sae

def get_config(path, idx):
    return json.load(open(os.path.join(path, F'sae_{idx}/config.json')))

def get_samples(model, dataloader, num_samples=128):
    template = dataloader.dataset.template
    dec = dataloader.dataset.decorator_length
    grammar = dataloader.dataset.PCFG
    inputs = template.repeat(num_samples, 1)
    samples, per_token_logprobs = model.sample(
        inputs=inputs, 
        max_new_tokens=118,
        retrieve_llhoods='tokens',
        )
    sentences = [grammar.detokenize_sentence(s[dec:]).split('<eos>')[0].strip() for s in samples]
    logprobs = [per_token_logprobs[i][dec:][:len(sentences[i].split())] for i in range(num_samples)]
    return sentences, logprobs

def get_latents_and_sequences(sae, dataloader, sae_dataloader, num_batches=128):
    latents = []
    sequences = []
    dec = dataloader.dataset.decorator_length
    pcfg = dataloader.dataset.PCFG
    pad = pcfg.vocab['<pad>']
    eos = pcfg.vocab['<eos>']

    for _ in range(num_batches):
        batch = next(iter(sae_dataloader))
        latent, _ = sae(batch[0])
        latents.append(latent.detach())
        sequences += [list(filter(lambda x : x not in [pad, eos], sequence)) for sequence in batch[1][:, dec:].tolist()]

    latents = torch.cat(latents, dim=0)

    return latents, sequences

def get_latents_by_sequence(latents, sequences):
    latents_by_sequence = []
    temp = latents.clone()
    for sequence in sequences:
        latents_by_sequence.append(temp[:len(sequence)])
        temp = temp[len(sequence):]
    return latents_by_sequence

parser = argparse.ArgumentParser()
# Define a parser that either accepts an "index" argument,
# or both "start" and "end" arguments.
parser.add_argument('--model_dir', type=str, required=True)
parser.add_argument('--start', type=int, required=True)
parser.add_argument('--end', type=int, required=False)
args = parser.parse_args()