import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


class LieGAN:
    """
    Lie group GAN for symmetry discovery

    Args:
        dim (int): Dimension of datapoints
        lie_dim (int): Dimension of Lie algebra
        sigma (float): Parameter controlling variance of generator
            multipliers
        form (string): Searchspace for symmetries, options are:
            'so' -> continuous rotations
            'se' -> continuous rotations and translations
            'dso' -> discrete and continuous rotations
            'dse' -> discrete and continuous rotations and translations
        device (string): Device to store tensors on
        seed (int): Seed for reproducability
        latent_dim (int): Dimension of latent space, None if no latent
            space used
        samp_dist (string): Sampling distribution for
        generator multipliers, options are:
            'normal'/'gaussian' -> w ~ N(0, sigma^2)
            'uniform' -> w ~ Uniform(-sigma, sigma)
        k_vals (string): Defines search space for period k, seperate by commas
            '3,4' would search over {1, 3, 4, c}
            '1:8' would search over {1, 2, 3, 4, 5, 6, 7, 8, c}
            '3:9::3' would search over {1, 3, 6, 9, c}
            '' would search over {1, c}
        tau (float): Temperature parameter for Gumbel Softmax distribution
        cloud_data (bool): True if datapoints are point clouds, False if
            datapoints are individual points
    """

    def __init__(self, dim, lie_dim, sigma=torch.pi,
                 form="so", device="cpu", seed=35,
                 latent_dim=None, samp_dist="normal",
                 k_vals="1:8", include_c=True, tau=2.0,
                 cloud_data=False):
        self.device = device
        if seed is not None:
            torch.manual_seed(seed)
        self.data = None
        self.loader = None
        self.dim = dim
        self.lie_dim = lie_dim
        self.form = form
        self.sigma = sigma
        self.latent_dim = latent_dim
        self.tau = tau
        self.cloud_data = cloud_data
        if latent_dim is not None:
            self.encoder = self.AutoEncoder(dim, latent_dim).to(device)
            self.algebra = self.LieAlgebra(latent_dim, lie_dim,
                                           form=form, k_vals=k_vals,
                                           include_c=include_c,
                                           device=device).to(device)
            self.g = self.Generator(self.algebra, sigma, form=form,
                                    samp_dist=samp_dist, tau=tau,
                                    device=device).to(device)
        else:
            self.algebra = self.LieAlgebra(self.dim, lie_dim,
                                           form=form, k_vals=k_vals,
                                           include_c=include_c,
                                           device=device).to(device)
            self.g = self.Generator(self.algebra, sigma,
                                    form=form, samp_dist=samp_dist,
                                    tau=tau, device=device).to(device)
        if cloud_data:
            self.d = self.CloudDiscriminator(self.dim).to(device)
        else:
            self.d = self.Discriminator(self.dim).to(device)

    class AutoEncoder(nn.Module):
        """
        Encode inputs to a latent representation and reconstruct them.

        Args:
            input_dim (int): Dimension of datapoints
            latent_dim (int): Dimension of latent space
        """

        def __init__(self, input_dim, latent_dim):
            super().__init__()
            self.linear1 = nn.Linear(input_dim, 32)
            self.linear2 = nn.Linear(32, 64)
            self.linear3 = nn.Linear(64, 32)
            self.linear4 = nn.Linear(32, latent_dim)
            self.linear5 = nn.Linear(latent_dim, 32)
            self.linear6 = nn.Linear(32, 64)
            self.linear7 = nn.Linear(64, 32)
            self.linear8 = nn.Linear(32, input_dim)

        def encode(self, x):
            x = F.relu(self.linear1(x))
            x = F.relu(self.linear2(x))
            x = F.relu(self.linear3(x))
            x = self.linear4(x)
            return x

        def decode(self, x):
            x = F.relu(self.linear5(x))
            x = F.relu(self.linear6(x))
            x = F.relu(self.linear7(x))
            x = self.linear8(x)
            return x

        def forward(self, x):
            z = self.encode(x)
            x_hat = self.decode(z)
            return x_hat

    def insert_data(self, data, batch_size=128):
        self.data = data.to("cpu")
        self.loader = torch.utils.data.DataLoader(self.data,
                                                  batch_size=batch_size,
                                                  shuffle=True)

    class Discriminator(nn.Module):
        """
        Classifies points as original data or symmetry-transformed data.

        Args:
            input_dim (int): Dimension of datapoints
        """

        def __init__(self, input_dim):
            super().__init__()
            self.linear1 = nn.Linear(input_dim, 32)
            self.linear2 = nn.Linear(32, 64)
            self.linear3 = nn.Linear(64, 128)
            self.linear4 = nn.Linear(128, 64)
            self.linear5 = nn.Linear(64, 1)
            self.sigmoid = nn.Sigmoid()

        def forward(self, x):
            x = F.leaky_relu(self.linear1(x))
            x = F.leaky_relu(self.linear2(x))
            x = F.leaky_relu(self.linear3(x))
            x = F.leaky_relu(self.linear4(x))
            x = self.sigmoid(self.linear5(x))
            return x

    class CloudDiscriminator(nn.Module):
        """
        Classifies point clouds as original data or symmetry-transformed data.

        Args:
            input_dim (int): Dimension of datapoints
        """

        def __init__(self, input_dim, hidden_dim=128):
            super().__init__()
            self.linear1 = nn.Linear(input_dim, hidden_dim)
            self.linear2 = nn.Linear(hidden_dim, hidden_dim)
            self.linear3 = nn.Linear(hidden_dim, hidden_dim)
            self.linear4 = nn.Linear(hidden_dim, hidden_dim)
            self.linear5 = nn.Linear(hidden_dim, hidden_dim)
            self.linear6 = nn.Linear(hidden_dim, 1)
            self.sigmoid = nn.Sigmoid()

        def forward(self, x):
            h = F.leaky_relu(self.linear1(x))
            h = F.leaky_relu(self.linear2(h))
            h = F.leaky_relu(self.linear3(h))
            h_max = h.max(dim=1).values
            g = F.leaky_relu(self.linear4(h_max))
            g = F.leaky_relu(self.linear5(g))
            return self.sigmoid(self.linear6(g))

    class LieAlgebra(nn.Module):
        """
        Lie algebra returning generator matrices.

        Args:
            dim (int): Dimension of datapoints
            lie_dim (int): Dimension of Lie algebra
            form (string): Searchspace for symmetries, options are:
                'so' -> continuous rotations
                'se' -> continuous rotations and translations
                'dso' -> discrete and continuous rotations
                'dse' -> discrete and continuous rotations and translations
            k_vals (string): Defines search space for period k, seperate
                using commas
                '3,4' would search over {1, 3, 4, c}
                '1:8,12' would search over {1, 2, 3, 4, 5, 6, 7, 8, 12, c}
                '3:9::3' would search over {1, 3, 6, 9, c}
                '' would search over {1, c}
            include_c (bool): True if continuous case is considered with
                symmetry candidates
            device (string): Device to store tensors on
        """

        def __init__(self, dim, lie_dim, form="so", k_vals="1:8",
                     include_c=True, device="cpu"):
            super().__init__()
            self.dim = dim
            self.form = form
            self.device = device
            self.include_c = include_c
            if form in ["so", "se"]:
                self.params = nn.Parameter(
                    torch.randn(lie_dim,
                                dim * (dim - 1) // 2).to(self.device) * 0.1)
                self.lie_dim = lie_dim
            elif form in ["dso", "dse"]:
                self.params = nn.Parameter(
                    torch.randn(lie_dim,
                                dim * (dim - 1) // 2).to(self.device) * 0.6)
                self.lie_dim = lie_dim
                k_strings = k_vals.split(",")
                k_space = set()
                for k_string in k_strings:
                    if "::" in k_string:
                        k_range, k_step = k_string.split("::")
                        if ":" in k_range:
                            for i in range(int(k_range.split(":")[0]),
                                           int(k_range.split(":")[1]) + 1,
                                           int(k_step)):
                                k_space.add(i)
                    elif ":" in k_string:
                        for i in range(int(k_string.split(":")[0]),
                                       int(k_string.split(":")[1]) + 1):
                            k_space.add(i)
                    else:
                        k_space.add(int(k_string))
                k_space.add(1)
                k_space = sorted(list(k_space))
                if include_c:
                    k_space.append("c")
                self.k_space = k_space
                self.k_logits = nn.Parameter(
                    torch.zeros(lie_dim,
                                len(k_space),
                                device=self.device))
            else:
                self.params = nn.Parameter(
                    torch.randn(lie_dim,
                                dim**2).to(self.device) * 0.1)
                self.lie_dim = lie_dim
            if form in ["se", "dse"]:
                self.c = torch.zeros(dim, device=self.device)

        def even(self, x):
            return 1 if x % 2 == 0 else -1

        def get_matrices(self):
            L_list = []
            if self.form in ["so", "dso"]:
                for i in range(self.lie_dim):
                    mat = torch.zeros(self.dim, self.dim, device=self.device)
                    idx = 0
                    for row in range(self.dim):
                        for col in range(row):
                            mat[row, col] = -self.even(idx) * self.params[i, idx]
                            mat[col, row] = self.even(idx) * self.params[i, idx]
                            idx += 1
                    L_list.append(mat)
            elif self.form in ["se", "dse"]:
                for i in range(self.lie_dim):
                    omega = torch.zeros(self.dim, self.dim,
                                        device=self.device)
                    idx = 0
                    for row in range(self.dim):
                        for col in range(row):
                            omega[row, col] = -self.even(idx) * self.params[i, idx]
                            omega[col, row] = self.even(idx) * self.params[i, idx]
                            idx += 1
                    mat = torch.zeros(self.dim + 1, self.dim + 1, device=self.device)
                    mat[0:self.dim, 0:self.dim] = omega
                    v = omega @ self.c
                    mat[0:self.dim, self.dim] = -v
                    L_list.append(mat)
            else:
                for i in range(self.lie_dim):
                    mat = torch.zeros(self.dim, self.dim, device=self.device)
                    for row in range(self.dim):
                        for col in range(self.dim):
                            mat[row, col] = self.params[i,
                                                        self.dim * row + col]
                    L_list.append(mat)
            return torch.stack(L_list)

    def set_centre(self, point):
        if self.form in ["se", "dse"]:
            self.algebra.c = point.to(self.device)

    class Generator(nn.Module):
        """
        Generator returning transformed samples.

        Args:
            lie_algebra (LieAlgebra): Stores matrix generators
            sigma (float): Parameter controlling variance of generator
            multipliers
            form (string): Searchspace for symmetries, options are:
                'so' -> continuous rotations
                'se' -> continuous rotations and translations
                'dso' -> discrete and continuous rotations
                'dse' -> discrete and continuous rotations and translations
            samp_dist (string): Sampling distribution for
                generator multipliers, options are:
                'normal'/'gaussian' -> w ~ N(0, sigma^2)
                'uniform' -> w ~ Uniform(-sigma, sigma)
            tau (float): Temperature parameter for Gumbel Softmax distribution
                device (string): Device to store tensors on
        """

        def __init__(self, lie_algebra, sigma,
                     form, samp_dist="normal",
                     tau=2.0, device="cpu"):
            super().__init__()
            self.lie_algebra = lie_algebra
            self.sigma = sigma
            self.form = form
            self.samp_dist = samp_dist
            self.tau = tau
            self.device = device

        def forward(self, x, j=0, k=1, sampling_size=2, cont_sampling_size=8):
            lie_basis = self.lie_algebra.get_matrices()
            if self.form == "se":
                if self.samp_dist in ["normal", "gaussian"]:
                    w = torch.randn(self.lie_algebra.lie_dim).to(self.device) * self.sigma
                else:
                    w = (2*torch.rand(self.lie_algebra.lie_dim).to(self.device) - 1) * self.sigma
                g = torch.eye(self.lie_algebra.dim + 1, device=self.device)
                for i in range(self.lie_algebra.lie_dim):
                    g = g @ torch.matrix_exp(w[i] * lie_basis[i])
                ones = torch.ones(x.shape[0], 1, device=self.device)
                xh = torch.cat([x, ones], dim=1)
                result = xh @ g.t()
                return result[:, :self.lie_algebra.dim]
            elif self.form == "dso":
                if k == 1:
                    return [x]
                elif k == "c":
                    sampled_w = 2 * torch.pi * torch.rand(cont_sampling_size).to(self.device)
                    G = [torch.matrix_exp(
                         (w / self.lie_algebra.params[j].detach().norm()) * lie_basis[j])
                         for w in sampled_w]
                    return [x @ g.t() for g in G]
                sampled_m = torch.randint(1, k, (sampling_size,)).to(self.device)
                G = [torch.matrix_exp(2 * m * (torch.pi / k) /
                     self.lie_algebra.params[j].detach().norm() *
                     lie_basis[j]) for m in sampled_m]
                return [x @ g.t() for g in G]
            elif self.form == "dse":
                if k == 1:
                    return [x]
                elif k == "c":
                    sampled_w = 2 * torch.pi * torch.rand(cont_sampling_size).to(self.device)
                    G = [torch.matrix_exp(
                         (w / self.lie_algebra.params[j].detach().norm()) * lie_basis[j])
                         for w in sampled_w]
                    if x.dim() == 2:
                        ones = torch.ones(x.shape[0], 1, device=self.device)
                        xh = torch.cat([x, ones], dim=1)
                    else:
                        ones = torch.ones(x.shape[0], x.shape[1], 1, device=self.device)
                        xh = torch.cat([x, ones], dim=2)
                    result = [(xh @ g.t())[..., :self.lie_algebra.dim] for g in G]
                    return result
                sampled_m = torch.randint(1, k, (sampling_size,)).to(self.device)
                G = [torch.matrix_exp(2 * m * (torch.pi / k) /
                     self.lie_algebra.params[j].detach().norm() *
                     lie_basis[j]) for m in sampled_m]
                if x.dim() == 2:
                    ones = torch.ones(x.shape[0], 1, device=self.device)
                    xh = torch.cat([x, ones], dim=1)
                else:
                    ones = torch.ones(x.shape[0], x.shape[1], 1, device=self.device)
                    xh = torch.cat([x, ones], dim=2)
                result = [(xh @ g.t())[..., :self.lie_algebra.dim] for g in G]
                return result
            else:
                if self.samp_dist in ["normal", "gaussian"]:
                    w = torch.randn(self.lie_algebra.lie_dim).to(self.device) * self.sigma
                else:
                    w = (2*torch.rand(self.lie_algebra.lie_dim).to(self.device) - 1) * self.sigma
                g = torch.eye(self.lie_algebra.dim, device=self.device)
                for i in range(self.lie_algebra.lie_dim):
                    g = g @ torch.matrix_exp(w[i] * lie_basis[i])
                return x @ g.t()

    def softmin(self, x):
        rho = 5.0
        return -(1 / rho) * torch.log(torch.mean(torch.exp(-rho * x)) + 1e-8)

    def train(self, epochs=60, dlr=1e-3, glr=1e-3, llr=2e-2,
              decay_lr=False, logit_epoch_split=0.6, lambda_o=0.004,
              lambda_r=0.005, lambda_c=0, lambda_k=0.005, lambda_e=0,
              print_losses=False, print_epochs=True, optimiser="Adam",
              m_sample_size=2, cont_sample_size=8, disc_samples=10,
              gumbel_hard=True):
        """
        Trains LieGAN.

        Args:
            epochs (int): Number of training runs through the dataset
            dlr (float): Discriminator learning rate
            glr (float): Generator learning rate
            llr (float): Period logit learning rate
            decay_lr (bool): If True, gradually decay generator learning rate
                to 0
            logit_epoch_split (float): Percentage of epochs before logit
                training starts
            lambda_o (float): Weight for orthogonality regularisation term.
                Note that 0.5 is recommended for discrete search
            lambda_r (float): Weight for trivial solution regularisation term
            lambda_c (float): Weight for reconstruction loss
            lambda_k (float): Weight for larger period bias regularisation term
            lambda_e (float): Weight for period entropy regularisation term
            print_losses (bool): If True, show losses each epoch
            print_epochs (bool): If True, show current epoch depending on
                total epochs
            optimiser (string): Select optimiser, options are:
                'Adam' - Adam optimiser
                'SGD' - Stochastic gradient descent
            m_sample_size (int): Number of angles sampled from some period
                during training for an individual period value
            cont_sample_size (int): Number of angles sampled for the continuous
                case during discrete rotation training
            disc_samples (int): Number of angles sampled from the set of all
                possible discrete angles during the discriminator step
            gumbel_hard (bool): If True, use straight-through trick on Gumbel
                Softmax
        """
        latent = self.latent_dim is not None
        mse = nn.MSELoss()
        if optimiser == "Adam":
            if not latent:
                if self.form in ["dso", "dse"]:
                    g_optim = optim.Adam([
                        {"params": [self.algebra.params], "lr": glr},
                        {"params": [self.algebra.k_logits], "lr": llr}
                    ])
                else:
                    g_optim = optim.Adam(self.g.parameters(), lr=glr)
            else:
                g_optim = optim.Adam(list(self.g.parameters()) +
                                     list(self.encoder.parameters()), lr=glr)
        elif optimiser == "SGD":
            if not latent:
                if self.form in ["dso", "dse"]:
                    g_optim = optim.SGD([
                        {"params": [self.algebra.params], "lr": glr},
                        {"params": [self.algebra.k_logits], "lr": llr}
                    ])
                else:
                    g_optim = optim.SGD(self.g.parameters(), lr=glr)
            else:
                g_optim = optim.SGD(list(self.g.parameters()) +
                                    list(self.encoder.parameters()), lr=glr)
        if decay_lr:
            _ = torch.optim.lr_scheduler.CosineAnnealingLR(g_optim,
                                                           T_max=epochs)
        if optimiser == "Adam":
            d_optim = optim.Adam(self.d.parameters(), lr=dlr)
        elif optimiser == "SGD":
            d_optim = optim.SGD(self.d.parameters(), lr=dlr)
        if self.form in ["dso", "dse"]:
            self.algebra.k_logits.requires_grad = False
            self.algebra.params.requires_grad = True
        for epoch in range(1, epochs + 1):
            if self.form in ["dso", "dse"] and epoch > epochs * logit_epoch_split:
                self.algebra.k_logits.requires_grad = True
                self.algebra.params.requires_grad = False
            for x in self.loader:
                x = x.to(self.device)
                if self.form in ["dso", "dse"]:
                    for i in range(self.lie_dim):
                        y = F.gumbel_softmax(self.algebra.k_logits[i],
                                             tau=self.tau, hard=gumbel_hard)
                        k_vals = self.algebra.k_space
                        angles = torch.tensor(
                            list(
                                {float(2 * np.pi * j / k)
                                 for k in k_vals[1:-1]
                                 for j in range(1, k)}
                            ), device=self.device)
                        idx = torch.randint(angles.shape[0], (disc_samples,)).to(self.device)
                        thetas = angles[idx]
                        lie_basis = self.algebra.get_matrices()
                        G = [torch.matrix_exp(
                             (theta / self.algebra.params[i].detach().norm()) * lie_basis[i])
                             for theta in thetas]
                        if self.form == "dso":
                            g_x = [x.detach() @ g.t() for g in G]
                        elif self.form == "dse":
                            if x.dim() == 2:
                                ones = torch.ones(x.shape[0], 1, device=self.device)
                                xh = torch.cat([x, ones], dim=1)
                            else:
                                ones = torch.ones(x.shape[0], x.shape[1], 1, device=self.device)
                                xh = torch.cat([x, ones], dim=2)
                            g_x = [(xh.detach() @ g.t())[..., :self.dim] for g in G]
                        d_x = self.d(x)
                        loss_d = torch.stack([-torch.mean(torch.log(d_x + 1e-8) +
                                              torch.log(1 - self.d(g_x_i) + 1e-8))
                                              for g_x_i in g_x]).mean()

                        # train discriminator
                        d_optim.zero_grad()
                        loss_d.backward()
                        d_optim.step()

                        # generator loss
                        g_optim.zero_grad()
                        if epoch > epochs * logit_epoch_split:
                            g_losses = []
                            for k in k_vals:
                                g_x = self.g(x, j=i, k=k,
                                             sampling_size=m_sample_size)
                                d_g = torch.stack(
                                    [self.d(g_x_i) for g_x_i in g_x]).mean()
                                g_losses.append(-torch.mean(torch.log(d_g + 1e-8)))
                            loss_g = (y * torch.stack(g_losses)).sum()
                        else:
                            g_losses = []
                            for k in k_vals:
                                g_x = self.g(x, j=i, k=k,
                                             sampling_size=m_sample_size)
                                d_g = torch.stack(
                                    [self.d(g_x_i) for g_x_i in g_x]).mean()
                                g_losses.append(-torch.mean(torch.log(d_g + 1e-8)))
                            loss_g = self.softmin(torch.stack(g_losses[1::]))

                        # large k regularisation
                        if self.algebra.include_c:
                            k_vals_numerical = torch.tensor(k_vals[0:-1] + [max(k_vals[0:-1]) + 1], device=self.device)
                        else:
                            k_vals_numerical = k_vals
                        loss_k = -lambda_k * (k_vals_numerical * y).sum()

                        # k entropy regularisation
                        p = torch.softmax(self.algebra.k_logits[i], dim=0)
                        loss_e = lambda_e * (p * torch.log(p)).sum()

                        # orthogonality regularisation
                        li = self.algebra.get_matrices()
                        loss_ortho = 0
                        if self.lie_dim > 1:
                            loss_ortho = lambda_o * torch.stack(
                                [self.algebra.params[i] @ self.algebra.params[j].t()
                                 for i in range(self.lie_dim) for j in range(i + 1, self.lie_dim)]
                            ).pow(2).mean()

                        # regularisation against trivial solution
                        loss_lreg = lambda_r * (self.algebra.params.norm(dim=1) - 1).pow(2).mean()

                        # train generator
                        loss_total = loss_g + loss_k + loss_ortho + loss_lreg + loss_e
                        loss_total.backward()
                        g_optim.step()
                else:

                    # train discriminator
                    d_optim.zero_grad()
                    if not latent:
                        g_x = self.g(x.detach())
                    else:
                        g_x = self.encoder.decode(
                            self.g(self.encoder.encode(x.detach())))
                    d_x = self.d(x)
                    d_g_x = self.d(g_x.detach())
                    loss_d = -torch.mean(
                        torch.log(d_x + 1e-8) + torch.log(1 - d_g_x + 1e-8))
                    loss_d.backward()
                    d_optim.step()

                    # generator loss
                    g_optim.zero_grad()
                    if not latent:
                        g_x = self.g(x)
                    else:
                        g_x = self.encoder.decode(
                            self.g(self.encoder.encode(x)))
                    d_g_x = self.d(g_x)
                    loss_g = -torch.mean(torch.log(d_g_x + 1e-8))

                    # orthogonality regularisation
                    li = self.algebra.get_matrices()
                    li_se = torch.stack([L[:-1, :-1] for L in li])
                    loss_ortho = 0
                    if self.lie_dim > 1:
                        if self.form == "so":
                            li_mat = torch.stack([x.reshape(-1) for x in li])
                            G = li_mat @ li_mat.t()
                            off_diag = G.triu(diagonal=1)
                            loss_ortho = (2 * off_diag).pow(2).sum()
                        elif self.form == "se":
                            li_se_mat = torch.stack(
                                [x.reshape(-1) for x in li_se])
                            G = li_se_mat @ li_se_mat.t()
                            off_diag = G.triu(diagonal=1)
                            loss_ortho = (2 * off_diag).pow(2).sum()
                        loss_ortho = loss_ortho / (self.lie_dim *
                                                   (self.lie_dim - 1) // 2)
                    loss_ortho = lambda_o * loss_ortho

                    # regularisation against trivial solution
                    loss_lreg = 0
                    if self.form == "so":
                        loss_lreg = (li.norm(dim=(1, 2)) - 1).pow(2).mean()
                    elif self.form == "se":
                        loss_lreg = (li_se.norm(dim=(1, 2)) - 1).pow(2).mean()
                    loss_lreg = lambda_r * loss_lreg

                    # reconstruction error
                    loss_con = 0
                    if latent:
                        x_hat = self.encoder(x)
                        loss_con = lambda_c * mse(x_hat, x)

                    # train generator
                    loss_total = loss_g + loss_ortho + loss_lreg + loss_con
                    loss_total.backward()
                    g_optim.step()

            if print_losses:
                if self.form in ["dso", "dse"]:
                    print(f"\nEpoch {epoch} | D loss: {loss_d:.4f} | " +
                          f"G loss: {loss_g:.4f} | K Loss: {loss_k:.4f} | " +
                          f"Ortho Loss: {loss_ortho:.4f} | LReg Loss: {loss_lreg:.4f}")
                    print("Lie Algebra:")
                    print(self.algebra.get_matrices())
                    print("Weights:")
                    print(torch.softmax(self.algebra.k_logits,
                                        dim=1).detach().round(decimals=3))
                elif latent:
                    print(f"\nEpoch {epoch} | D loss: {loss_d:.4f} | " +
                          f"G loss: {loss_g:.4f} | Ortho Loss: " +
                          f"{loss_ortho:.4f} | LReg Loss: {loss_lreg:.4f}" +
                          f" | Recon Loss: {loss_con.item():.4f}")
                    print("Lie Algebra:")
                    print(self.algebra.get_matrices())
                else:
                    print(f"\nEpoch {epoch} | D loss: {loss_d:.4f} | " +
                          f"G loss: {loss_g:.4f} | Ortho Loss: " +
                          f"{loss_ortho:.4f} | LReg Loss: {loss_lreg:.4f}")
                    print("Lie Algebra:")
                    print(self.algebra.get_matrices())
                    print(f"d(x) = {d_x.detach().mean()}")
                    print(f"d(g(x)) = {d_g_x.detach().mean()}")
            elif print_epochs:
                if epochs < 15:
                    print(f"Epoch {epoch}")
                elif epochs < 30 and epoch % 5 == 0:
                    print(f"Epoch {epoch}")
                elif epochs < 150 and epoch % 10 == 0:
                    print(f"Epoch {epoch}")
                elif epoch % 50 == 0:
                    print(f"Epoch {epoch}")
        print("Training Complete")

    def apply_transformation(self, w, scale_angles=False):
        """
        Apply learnt continuous transformation

        Args:
        w ([float]): Magnitude of rotations
        scale_angles (bool): Scale magnitudes to radian angles
        """
        lie = self.algebra.get_matrices()
        g = torch.eye(self.dim, device=self.device)
        for i in range(len(lie)):
            if scale_angles:
                w[i] = w[i] / self.algebra.params[i].norm()
            g = g @ torch.matrix_exp(w[i] * lie[i]).detach()
        return self.data @ g.t()

    def apply_latent_transformation(self, w, scale_angles=False):
        """
        Apply learnt continuous transformation in latent space

        Args:
        w ([float]): Magnitude of rotations
        scale_angles (bool): Scale magnitudes to radian angles
        """
        lie = self.algebra.get_matrices()
        g = torch.eye(self.latent_dim, device=self.device)
        for i in range(len(lie)):
            if scale_angles:
                w[i] = w[i] / self.algebra.params[i].norm()
            g = g @ torch.matrix_exp(w[i] * lie[i]).detach()
        z = self.encoder.encode(self.data)
        return self.encoder.decode(z @ g.t())

    def rotation_axes(self):
        """
        Return learnt rotational axes.
        """
        L = self.algebra.get_matrices()
        axes = []
        with torch.no_grad():
            for i in range(self.lie_dim):
                M = L[i][:self.dim, :self.dim]
                _, S, V = torch.svd(M)
                axis = V[:, -1]
                axes.append(axis)
        return axes
