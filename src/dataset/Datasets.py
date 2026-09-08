import torch
from torch.utils.data import Dataset
import numpy as np


class LorenzDataset(Dataset):

    def __init__(self, args, data_path):
        data = np.load(data_path, allow_pickle=True).item()
        self.x = torch.tensor(data['x']).view(-1, args.timesteps, args.u_dim)
        self.dx = torch.tensor(data['dx']).view(-1, args.timesteps, args.u_dim)
        self.dz = torch.tensor(data['dz']).view(-1, args.timesteps, args.z_dim)

    def __len__(self):
        return self.x.size(0)
    
    def __getitem__(self, idx):
        return self.x[idx], self.dx[idx], self.dz[idx]


class PendulumDataset(Dataset):

    def __init__(self, args, data_path):
        data = np.load(data_path, allow_pickle=True).item()
        self.x = torch.tensor(data['x']).view(-1, args.timesteps, args.u_dim)
        self.dx = torch.tensor(data['dx']).view(-1, args.timesteps, args.u_dim)
        self.dz = torch.tensor(data['dz']).view(-1, args.timesteps, args.z_dim)

    def __len__(self):
        return self.x.size(0)

    def __getitem__(self, idx):
        return self.x[idx], self.dx[idx], self.dz[idx]


class SynthDataset(Dataset):
    """'Coordinate problem' dataset: a low-D latent observed through an unknown
    nonlinear map. Used for `synth` (2D Duffing, src/dataset/synth_functions.py),
    `lorenz_distort` (3D Lorenz, src/dataset/lorenz_functions.py), and `cylinder`
    (real CFD pressure probes, create_cylinder.py -- no known
    ground-truth latent, so no 'z'/'dz' key; SINDyAE only ever consumes (x, dx)
    -- dz is computed on the fly from dx inside the model -- so a zero
    placeholder here is inert, just satisfying the loader's tuple shape)."""

    def __init__(self, args, data_path):
        data = np.load(data_path, allow_pickle=True).item()
        self.x = torch.tensor(data['x']).view(-1, args.timesteps, args.u_dim)
        self.dx = torch.tensor(data['dx']).view(-1, args.timesteps, args.u_dim)
        if 'dz' in data:
            if data['dz'].shape[-1] != args.z_dim:
                raise ValueError(f"{data_path} has z_dim={data['dz'].shape[-1]} but -Z {args.z_dim} "
                                 f"was passed (synth Duffing is 2D; lorenz_distort is 3D)")
            self.dz = torch.tensor(data['dz']).view(-1, args.timesteps, args.z_dim)
        else:
            self.dz = torch.zeros(self.x.size(0), args.timesteps, args.z_dim, dtype=self.x.dtype)

    def __len__(self):
        return self.x.size(0)

    def __getitem__(self, idx):
        return self.x[idx], self.dx[idx], self.dz[idx]