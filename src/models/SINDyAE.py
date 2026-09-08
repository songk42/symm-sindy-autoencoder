import torch
import torch.nn as nn
from src.utils.model_utils import library_size, sindy_library
from src.utils.symmetry import (InducedLibraryRep, so_matrix, pi_rotation_params,
                                equivariance_residual, finite_order_residual,
                                nontriviality_penalty)


class Sine(nn.Module):
    """sin activation (SIREN-style), analytic everywhere."""
    def forward(self, x):
        return torch.sin(x)


# activations that are real-analytic (or at least C^1 with a clean closed-form
# derivative) -- the coordinate problem needs the encoder to be a smooth
# immersion, so ReLU/LeakyReLU are deliberately excluded from the recommended set
_ANALYTIC = {'tanh', 'sin', 'sig', 'elu'}


class Net(nn.Module):

    def __init__(self, args):
        super(Net, self).__init__()

        self.z_dim = args.z_dim
        self.u_dim = args.u_dim
        self.hidden_dim = args.hidden_dim
        self.poly_order = args.poly_order
        self.use_sine = args.use_sine
        self.include_constant = args.include_constant
        self.library_dim = library_size(self.z_dim, self.poly_order, use_sine=self.use_sine, include_constant=self.include_constant)
        self.mse = nn.MSELoss(reduction='mean')
        self.nonlinearity = args.nonlinearity
        # spectral norm is a bi-Lipschitz control on the *coordinate map*, so it
        # is applied to the encoder only -- the decoder needs the freedom to
        # expand back onto the observation manifold
        self.spectral_norm = getattr(args, 'spectral_norm', 0)

        if self.nonlinearity is not None and self.nonlinearity not in _ANALYTIC \
                and self.nonlinearity != 'relu':
            raise ValueError(f"unknown nonlinearity {self.nonlinearity!r}")

        # psi^-1 can be (near-)linear on the observation manifold; a linear
        # encoder then IS the true chart up to a linear gauge, with no residual
        # nonlinear gauge freedom. The decoder still needs the nonlinearity to
        # expand back onto the observation manifold.
        self.linear_encoder = bool(getattr(args, 'linear_encoder', 0))
        self.encoder = self.build_net(self.u_dim, self.hidden_dim, self.z_dim,
                                      spectral=bool(self.spectral_norm),
                                      linear=self.linear_encoder)
        self.decoder = self.build_net(self.z_dim, self.hidden_dim, self.u_dim,
                                      spectral=False)
        self.sindy_coefficients = nn.Parameter(torch.randn(self.library_dim, self.z_dim, requires_grad=True))
        nn.init.xavier_normal_(self.sindy_coefficients)
        self.sequential_threshold = args.sequential_threshold
        self.threshold_mask = nn.Parameter(torch.ones_like(self.sindy_coefficients), requires_grad=False)

        # ---- kinematic-row constraint (2nd-order mechanical systems only) ----
        # Pin column 0 of Xi -- the RHS of zdot_0 -- to the trivial first-order
        # reduction identity  zdot_0 = z_1  (a one-hot on the 'z2' library term).
        # This removes the odd gauge freedom on the position coordinate that
        # leaves Z2-equivariance intact but warps the vector field. Invalid for
        # systems whose state is not a position/velocity pair (e.g. Lorenz).
        self.kinematic_row = bool(getattr(args, 'kinematic_row', 0))
        self._kin_col_idx = (1 if self.include_constant else 0) + 1   # index of z_1 in Theta

        # Jacobian-non-constancy penalty config (see _jac_loss): pushes E toward
        # affine, attacking the residual nonlinear gauge freedom.
        self.jac_probes = int(getattr(args, 'jac_probes', 4))

        # ---- Phase C: discrete-symmetry generators T_k in SO(d) ----
        # In 2D the only non-trivial order-2 element of SO(2) is -I, so "learning"
        # the generator is degenerate and the ||T^p - I||^2 penalty has a spurious
        # attractor at T = I (a barrier separates it from -I). --freeze_generators
        # pins T at the canonical pi-rotation and lets the *encoder* be the free
        # variable that has to make -I a symmetry -- the actual coordinate problem.
        self.n_generators = int(getattr(args, 'n_generators', 0))
        self.symmetry_order = int(getattr(args, 'symmetry_order', 2))
        self.freeze_generators = bool(getattr(args, 'freeze_generators', 0))
        if self.n_generators > 0:
            if self.use_sine:
                raise ValueError("n_generators > 0 needs a polynomial library "
                                 "(set use_sine=False): sin(Tz) is not a linear "
                                 "image of the library")
            if self.freeze_generators:
                A0 = pi_rotation_params(self.z_dim).expand(self.n_generators, -1, -1)
                self.gen_params = nn.Parameter(A0.clone(), requires_grad=False)
            else:
                gi = float(getattr(args, 'gen_init_scale', 1.0))
                self.gen_params = nn.Parameter(gi * torch.randn(self.n_generators,
                                                                self.z_dim, self.z_dim))
            self._induced_rep = InducedLibraryRep(
                self.z_dim, self.poly_order, include_constant=self.include_constant,
                use_sine=False)
        else:
            self.register_parameter('gen_params', None)
            self._induced_rep = None

    # ------------------------------------------------------------------ #
    def effective_xi(self):
        """The SINDy coefficient matrix actually used for prediction and for the
        equivariance residual: threshold-masked, with column 0 pinned to
        zdot_0 = z_1 when --kinematic_row is set."""
        Xi = self.sindy_coefficients * self.threshold_mask
        if self.kinematic_row:
            kin = Xi.new_zeros(Xi.shape[0])
            kin[self._kin_col_idx] = 1.0
            Xi = torch.cat([kin.unsqueeze(1), Xi[:, 1:]], dim=1)
        return Xi

    # ------------------------------------------------------------------ #
    def symmetry_generators(self):
        """List of the current generator matrices T_k in SO(d)."""
        if self.n_generators == 0:
            return []
        return [so_matrix(self.gen_params[k]) for k in range(self.n_generators)]

    def _equiv_loss(self, lambda_equiv, lambda_order, lambda_repel):
        """Phase C generator losses: intertwining residual + finite-order +
        anti-triviality, averaged over generators."""
        if self.n_generators == 0:
            return self.sindy_coefficients.new_zeros(())
        device = self.sindy_coefficients.device
        if self._induced_rep.Z.device != device:
            self._induced_rep.to(device)
        Xi = self.effective_xi()
        total = self.sindy_coefficients.new_zeros(())
        for T in self.symmetry_generators():
            total = total + lambda_equiv * equivariance_residual(Xi, T, self._induced_rep)
            if not self.freeze_generators:
                # only meaningful when T is being learned
                total = total + lambda_order * finite_order_residual(T, self.symmetry_order)
                total = total + lambda_repel * nontriviality_penalty(T)
        return total / self.n_generators

    # ------------------------------------------------------------------ #
    def _jac_loss(self, x, lambda_jac):
        """Relative batch variance of the encoder Jacobian's action on random
        probe directions:  mean_v  Var_i[ J_E(x_i) v ] / mean_i ||J_E(x_i) v||^2.

        Zero iff J_E is constant across the batch, i.e. E is affine. This is the
        lever against the odd-diffeomorphism gauge freedom that a linear
        symmetry action (or the kinematic-row constraint) leaves intact:
        f(T z) = T f(z) and zdot_0 = z_1 are both preserved by z1 -> phi(z1) for
        any odd phi, but such a phi makes J_E state-dependent.
        """
        if lambda_jac == 0 or x.shape[0] < 2:
            return x.new_zeros(())
        B, u = x.shape
        total = x.new_zeros(())
        for _ in range(self.jac_probes):
            v = torch.randn(u, device=x.device)
            v = v / (v.norm() + 1e-12)
            g = self.get_derivative(x, v.expand(B, u), self.encoder)   # (B, d)
            gc = g - g.mean(dim=0, keepdim=True)
            total = total + gc.pow(2).mean() / (g.pow(2).mean() + 1e-8)
        return lambda_jac * total / self.jac_probes

    # ------------------------------------------------------------------ #
    def encode(self, x):
        return self.encoder(x)

    def forward(self, x, dx, lambdas):
        batch_size, T, _ = x.shape
        device = self.sindy_coefficients.device

        # reshape data to be (b * t) x u
        x = x.reshape(-1, self.u_dim).type(torch.FloatTensor).to(device)
        dx = dx.reshape(-1, self.u_dim).type(torch.FloatTensor).to(device)

        # encode and decode
        z = self.encoder(x)
        x_recon = self.decoder(z)

        # build the SINDy library using the latent vector
        theta = sindy_library(z, self.poly_order, device, self.use_sine, self.include_constant)

        # calculate the z derivative
        dz = self.get_derivative(x, dx, self.encoder)

        # predict the z derivative using the library
        dz_pred = self.predict(theta)

        # predict the derivative of dx by using the predicted z derivative
        dx_pred = self.get_derivative(z, dz_pred, self.decoder)

        # calculate loss
        return self.loss_func(x, x_recon, dx_pred, dz_pred, dx, dz, z, lambdas)

    def predict(self, theta):
        # sindy_coefficients: L x z
        theta = theta.unsqueeze(1) # (b * T) x L  --->   (b * T) x 1 x L
        return torch.matmul(theta, self.effective_xi()).squeeze() # (b x T) x z

    # ------------------------------------------------------------------ #
    # Returns the first order time derivative of z (dz/dt) or the reconstructed
    # x (dx/dt), by propagating the input derivative through the network.
    def get_derivative(self, layer_output, dL, net):
        # a net with no activation modules (e.g. --linear_encoder) is affine, so
        # every activation derivative is 1 regardless of self.nonlinearity
        net_is_linear = all(isinstance(m, nn.Linear) for m in net)
        dz = dL
        for i in range(len(net) - 1):
            curr_layer = net[i]

            # if linear layer, get transposed weights and bias
            if isinstance(curr_layer, nn.Linear):
                wT, b = curr_layer.weight.T, curr_layer.bias
            else: # if its the activation function, skip to next layer
                continue

            # affine transformation before the activation function
            if self.nonlinearity is not None and not net_is_linear:
                output_before_act = torch.matmul(layer_output, wT) + b

            if net_is_linear:
                d_layer_output = 1

            elif self.nonlinearity == 'sig':
                # d/dx sigmoid(x) = s(x) (1 - s(x))
                layer_output = torch.sigmoid(output_before_act)
                d_layer_output = layer_output * (1 - layer_output)

            elif self.nonlinearity == 'relu':
                # d/dx relu(x) = 1[x > 0]
                layer_output = nn.functional.relu(output_before_act)
                d_layer_output = (output_before_act > 0).float()

            elif self.nonlinearity == 'elu':
                # d/dx elu(x) with alpha=1 = 1 if x > 0 else exp(x)
                layer_output = nn.functional.elu(output_before_act)
                d_layer_output = torch.minimum(torch.exp(output_before_act),
                                               torch.ones_like(output_before_act))

            elif self.nonlinearity == 'tanh':
                # d/dx tanh(x) = 1 - tanh(x)^2   (analytic)
                layer_output = torch.tanh(output_before_act)
                d_layer_output = 1 - layer_output ** 2

            elif self.nonlinearity == 'sin':
                # d/dx sin(x) = cos(x)           (analytic)
                layer_output = torch.sin(output_before_act)
                d_layer_output = torch.cos(output_before_act)

            else:
                # no activation function
                d_layer_output = 1
            dz = d_layer_output * torch.matmul(dz, wT)
        return torch.matmul(dz, net[-1].weight.T)

    def get_derivative_order2(self, layer_output, dL, ddL, net):
        """First and second order time derivatives propagated through the net."""
        net_is_linear = all(isinstance(m, nn.Linear) for m in net)
        dz = dL
        ddz = ddL
        for i in range(len(net) - 1):
            curr_layer = net[i]

            if isinstance(curr_layer, nn.Linear):
                wT, b = curr_layer.weight.T, curr_layer.bias
            else:
                continue

            if self.nonlinearity is not None and not net_is_linear:
                output_before_act = torch.matmul(layer_output, wT) + b

            if net_is_linear:
                d_layer_output = 1
                dd_layer_output = 1
                dz = torch.matmul(dz, wT)
                ddz = torch.matmul(ddz, wT)

            elif self.nonlinearity == 'sig':
                dz_prev = torch.matmul(dz, wT)
                layer_output = torch.sigmoid(output_before_act)
                d_layer_output = layer_output * (1 - layer_output)
                dd_layer_output = d_layer_output * (1 - 2 * layer_output)
                dz = d_layer_output * dz_prev
                ddz = (dd_layer_output * (dz_prev ** 2)) + (d_layer_output * torch.matmul(ddz, wT))

            elif self.nonlinearity == 'relu':
                layer_output = nn.functional.relu(output_before_act)
                d_layer_output = (output_before_act > 0).float()
                dz = d_layer_output * torch.matmul(dz, wT)
                ddz = d_layer_output * torch.matmul(ddz, wT)

            elif self.nonlinearity == 'elu':
                dz_prev = torch.matmul(dz, wT)
                layer_output = nn.functional.elu(output_before_act)
                d_layer_output = torch.minimum(torch.exp(output_before_act),
                                               torch.ones_like(output_before_act))
                dd_layer_output = torch.exp(output_before_act) * (output_before_act < 0).float()
                dz = d_layer_output * dz_prev
                ddz = (dd_layer_output * (dz_prev ** 2)) + (d_layer_output * torch.matmul(ddz, wT))

            elif self.nonlinearity == 'tanh':
                dz_prev = torch.matmul(dz, wT)
                layer_output = torch.tanh(output_before_act)
                d_layer_output = 1 - layer_output ** 2
                dd_layer_output = -2 * layer_output * d_layer_output
                dz = d_layer_output * dz_prev
                ddz = (dd_layer_output * (dz_prev ** 2)) + (d_layer_output * torch.matmul(ddz, wT))

            elif self.nonlinearity == 'sin':
                dz_prev = torch.matmul(dz, wT)
                layer_output = torch.sin(output_before_act)
                d_layer_output = torch.cos(output_before_act)
                dd_layer_output = -torch.sin(output_before_act)
                dz = d_layer_output * dz_prev
                ddz = (dd_layer_output * (dz_prev ** 2)) + (d_layer_output * torch.matmul(ddz, wT))

            else:
                d_layer_output = 1
                dd_layer_output = 1
                dz = torch.matmul(dz, wT)
                ddz = torch.matmul(ddz, wT)

        dz = d_layer_output * torch.matmul(dz, wT)
        ddz = dd_layer_output * torch.matmul(ddz, wT)
        return dz, ddz

    # ------------------------------------------------------------------ #
    def loss_func(self, x, x_recon, dx_pred, dz_pred, dx, dz, z, lambdas):
        (lambda_dx, lambda_dz, lambda_reg, lambda_gauge,
         lambda_equiv, lambda_order, lambda_repel, lambda_jac) = lambdas

        # reconstruction loss
        l_recon = self.mse(x_recon, x)

        # SINDy loss in dx
        l_dx = lambda_dx * self.mse(dx_pred, dx)

        # SINDy loss in dz
        l_dz = lambda_dz * self.mse(dz_pred, dz)

        # SINDy regularization
        l_reg = lambda_reg * torch.abs(self.sindy_coefficients).mean()

        # gauge regularization: pull the latent covariance towards the identity.
        # removes the scale/rotation ambiguity of the discovered coordinates and
        # guards against dimensional collapse of the latent state.
        if lambda_gauge != 0 and z.shape[0] > 1:
            zc = z - z.mean(dim=0, keepdim=True)
            cov = (zc.t() @ zc) / (z.shape[0] - 1)
            l_gauge = lambda_gauge * ((cov - torch.eye(self.z_dim, device=z.device)) ** 2).mean()
        else:
            l_gauge = z.new_zeros(())

        # Phase C: learned-generator equivariance (intertwining + order + repel)
        l_equiv = self._equiv_loss(lambda_equiv, lambda_order, lambda_repel)

        # coordinate-map linearity: penalize non-constant encoder Jacobian
        l_jac = self._jac_loss(x, lambda_jac)

        kl = l_recon.new_zeros(())

        return l_recon, l_dx, l_dz, l_reg, l_gauge, l_equiv, l_jac, kl

    # ------------------------------------------------------------------ #
    def _maybe_spectral_norm(self, linear):
        """Spectral normalization caps the largest singular value of each weight
        at 1, giving the encoder/decoder a controlled Lipschitz constant
        (bi-Lipschitz control for the coordinate map)."""
        if self.spectral_norm:
            return nn.utils.parametrizations.spectral_norm(linear)
        return linear

    def build_net(self, in_dim, hidden_dim, out_dim, spectral=False, linear=False):
        act = None if linear else {
            'elu': nn.ELU, 'sig': nn.Sigmoid, 'relu': nn.ReLU,
            'tanh': nn.Tanh, 'sin': Sine,
        }.get(self.nonlinearity)
        sn = self._maybe_spectral_norm if spectral else (lambda m: m)
        if act is None:  # linear autoencoder
            return nn.Sequential(
                sn(nn.Linear(in_dim, hidden_dim)),
                sn(nn.Linear(hidden_dim, hidden_dim)),
                sn(nn.Linear(hidden_dim, out_dim)),
            )
        return nn.Sequential(
            sn(nn.Linear(in_dim, hidden_dim)),
            act(),
            sn(nn.Linear(hidden_dim, hidden_dim)),
            act(),
            sn(nn.Linear(hidden_dim, out_dim)),
        )
