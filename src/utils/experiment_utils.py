import torch
import numpy as np
from src.utils.model_utils import init_weights, equation_sindy_library, get_equation

def print_gov_eqs(net):
    library = equation_sindy_library(net.z_dim, net.poly_order)
    xi = net.effective_xi() if hasattr(net, "effective_xi") \
        else net.threshold_mask * net.sindy_coefficients
    coefs = xi.detach().cpu().numpy()
    names = ['X', 'Y', 'Z'] + [f'Z{i}' for i in range(4, net.z_dim + 1)]
    for i in range(net.z_dim):
        print(get_equation(library, coefs[:, i], f"{names[i]}' = "))