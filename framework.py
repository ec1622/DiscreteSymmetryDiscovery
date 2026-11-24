import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import math


class Model:
    """
    Model for discrete symmetry discovery in point cloud data

    Args:
        dim (int): Dimension of datapoints
        n_gen (int): Number of generators
        form (string): Lie algebra parametrisation, options are:
            'dso' -> zero-centre rotations
            'dse' -> off-centre rotations
        device (string): Device to store tensors on
        seed (int): Seed for reproducibility
        k_vals (string): Defines search space for period k, separate using commas
            '3,4' would search over {1, 3, 4}
            '1:8,12' would search over {1, 2, 3, 4, 5, 6, 7, 8, 12}
            '3:9::3' would search over {1, 3, 6, 9}
            '' would search over {1}
        cloud_data (bool): True if datapoints are point clouds, False if datapoints are individual points
    """

    def __init__(self, dim, n_gen, form="dso", device="cpu", seed=None, k_vals="1:10", cloud_data=True):
        self.device = device
        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        self.data = None
        self.loader = None
        self.dim = dim
        self.n_gen = n_gen
        self.form = form
        self.cloud_data = cloud_data
        self.algebra = self.LieAlgebra(self.dim, n_gen, form=form, k_vals=k_vals, device=device).to(device)
        self.g = self.Generator(self.algebra, form=form, device=device).to(device)
        if cloud_data:
            self.d = self.CloudDiscriminator(self.dim).to(device)
        else:
            self.d = self.Discriminator(self.dim).to(device)

    def insert_data(self, data, batch=True, batch_size=3):
        self.data = data.to(self.device)
        if not batch:
            if self.cloud_data:
                full = self.data.unsqueeze(0)
            else:
                full = self.data
            self.loader = lambda: [full]
        else:
            if self.cloud_data:
                self.batch_size = batch_size
                self.loader = self.cloud_batches
            else:
                batch_loader = torch.utils.data.DataLoader(self.data, batch_size=batch_size, shuffle=True)
                self.loader = lambda: batch_loader

    def cloud_batches(self):
        N = self.data.shape[0]
        n = N // self.batch_size
        perm = torch.randperm(N, device=self.device)[:self.batch_size * n]
        yield self.data[perm].reshape(self.batch_size, n, -1)

    class Discriminator(nn.Module):
        """
        Classifies points as original data or symmetry-transformed data.

        Args:
            input_dim (int): Dimension of datapoints
        """

        def __init__(self, input_dim):
            super().__init__()
            self.linear1 = nn.Linear(input_dim, 64)
            self.linear2 = nn.Linear(64, 128)
            self.linear3 = nn.Linear(128, 128)
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
            hidden_dim (int): Number of dimensions in hidden layers
        """

        def __init__(self, input_dim, hidden_dim=128):
            super().__init__()
            self.linear1 = nn.Linear(input_dim, hidden_dim)
            self.linear2 = nn.Linear(hidden_dim, hidden_dim)
            self.linear3 = nn.Linear(hidden_dim, hidden_dim)
            self.linear4 = nn.Linear(2 * hidden_dim, 2 * hidden_dim)
            self.linear5 = nn.Linear(2 * hidden_dim, 2 * hidden_dim)
            self.linear6 = nn.Linear(2 * hidden_dim, 1)
            self.sigmoid = nn.Sigmoid()

        def forward(self, x):
            h = F.leaky_relu(self.linear1(x))
            h = F.leaky_relu(self.linear2(h))
            h = F.leaky_relu(self.linear3(h))
            h_max = h.max(dim=1).values
            h_mean = h.mean(dim=1)
            h_x = torch.cat([h_max, h_mean], dim=1)
            g = F.leaky_relu(self.linear4(h_x))
            g = F.leaky_relu(self.linear5(g))
            return self.sigmoid(self.linear6(g))

    class LieAlgebra(nn.Module):
        """
        Lie algebra returning generator matrices.

        Args:
            dim (int): Dimension of datapoints
            n_gen (int): Number of generators
            form (string): Lie algebra parametrisation, options are:
                'dso' -> zero-centre rotations
                'dse' -> off-centre rotations
            k_vals (string): Defines search space for period k, separate using commas
                '3,4' would search over {1, 3, 4}
                '1:8,12' would search over {1, 2, 3, 4, 5, 6, 7, 8, 12}
                '3:9::3' would search over {1, 3, 6, 9}
                '' would search over {1}
            device (string): Device to store tensors on
        """

        def __init__(self, dim, n_gen, form="dso", k_vals="1:10", device="cpu"):
            super().__init__()
            self.dim = dim
            self.form = form
            self.device = device
            self.params = nn.Parameter(torch.randn(n_gen, dim * (dim - 1) // 2).to(self.device) * 0.6)
            self.n_gen = n_gen
            k_strings = k_vals.split(",")
            k_space = set()
            for k_string in k_strings:
                if "::" in k_string:
                    k_range, k_step = k_string.split("::")
                    if ":" in k_range:
                        for i in range(int(k_range.split(":")[0]), int(k_range.split(":")[1]) + 1, int(k_step)):
                            k_space.add(i)
                elif ":" in k_string:
                    for i in range(int(k_string.split(":")[0]), int(k_string.split(":")[1]) + 1):
                        k_space.add(i)
                else:
                    k_space.add(int(k_string))
            k_space.add(1)
            k_space = sorted(list(k_space))
            self.k_space = k_space
            self.k_logits = nn.Parameter(
                torch.zeros(n_gen, len(k_space), device=self.device))
            if form == "dse":
                self.c = torch.zeros(dim, device=self.device)

        def even(self, x):
            return 1 if x % 2 == 0 else -1

        def get_matrices(self):
            L_list = []
            if self.form == "dso":
                for i in range(self.n_gen):
                    mat = torch.zeros(self.dim, self.dim, device=self.device)
                    idx = 0
                    for row in range(self.dim):
                        for col in range(row):
                            mat[row, col] = -self.even(idx) * self.params[i, idx]
                            mat[col, row] = self.even(idx) * self.params[i, idx]
                            idx += 1
                    L_list.append(mat)
            elif self.form == "dse":
                for i in range(self.n_gen):
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
            return torch.stack(L_list)

    def set_centre(self, point):
        if self.form == "dse":
            self.algebra.c = point.to(self.device)

    class Generator(nn.Module):
        """
        Generator returning transformed samples.

        Args:
            lie_algebra (LieAlgebra): Stores matrix generators
            form (string): Lie algebra parametrisation, options are:
                'dso' -> zero-centre rotations
                'dse' -> off-centre rotations
            device (string): Device to store tensors on
        """

        def __init__(self, lie_algebra, form, device="cpu"):
            super().__init__()
            self.lie_algebra = lie_algebra
            self.form = form
            self.device = device

        def forward(self, x, j=0, k=1, sampling_size=2):
            lie_basis = self.lie_algebra.get_matrices()
            if self.form == "dso":
                if k == 1:
                    return [x]
                sampled_m = torch.randint(1, k, (sampling_size,)).to(self.device)
                G = [torch.matrix_exp(2 * m * (torch.pi / k) * lie_basis[j]) for m in sampled_m]
                return [x @ g.t() for g in G]
            elif self.form == "dse":
                if k == 1:
                    return [x]
                sampled_m = torch.randint(1, k, (sampling_size,)).to(self.device)
                G = [torch.matrix_exp(2 * m * (torch.pi / k) * lie_basis[j]) for m in sampled_m]
                if x.dim() == 2:
                    ones = torch.ones(x.shape[0], 1, device=self.device)
                    xh = torch.cat([x, ones], dim=1)
                else:
                    ones = torch.ones(x.shape[0], x.shape[1], 1, device=self.device)
                    xh = torch.cat([x, ones], dim=2)
                result = [(xh @ g.t())[..., :self.lie_algebra.dim] for g in G]
                return result

    def smoothmin(self, x, rho=3.0):
        return -(1 / rho) * torch.log(torch.mean(torch.exp(-rho * x)) + 1e-8)

    def train(self, epochs=200, dlr=1e-3, glr=4e-2, llr=2e-2,
              logit_epoch_split=0.5, lambda_o=0.005, lambda_e=0.05,
              print_losses=False, print_epochs=True, optimiser="Adam",
              rho=3.0, tau=2.0, m_sample_size=6, disc_samples=6):
        """
        Trains framework.

        Args:
            epochs (int): Number of training runs through the dataset
            dlr (float): Discriminator learning rate
            glr (float): Generator learning rate
            llr (float): Period logit learning rate
            logit_epoch_split (float): Fraction of epochs before logit training starts
            lambda_o (float): Weight for orthogonality regularisation term
            lambda_e (float): Weight for period entropy regularisation term
            print_losses (bool): If True, show losses each epoch
            print_epochs (bool): If True, show current epoch depending on total epochs
            optimiser (string): Select optimiser, options are:
                'Adam' - Adam optimiser
                'SGD' - Stochastic gradient descent
            rho (float): Smoothmin temperature parameter
            tau (float): Softmax temperature parameter
            m_sample_size (int): Number of angles sampled from some period during training for an individual period value
            disc_samples (int): Number of angles sampled from the set of all possible discrete angles during the discriminator step
        """
        if optimiser == "Adam":
            g_optim = optim.Adam([{"params": [self.algebra.params], "lr": glr}, {"params": [self.algebra.k_logits], "lr": llr}])
        elif optimiser == "SGD":
            g_optim = optim.SGD([{"params": [self.algebra.params], "lr": glr}, {"params": [self.algebra.k_logits], "lr": llr}])
        if optimiser == "Adam":
            d_optim = optim.Adam(self.d.parameters(), lr=dlr, betas=(0.5, 0.999))
        elif optimiser == "SGD":
            d_optim = optim.SGD(self.d.parameters(), lr=dlr)
        self.algebra.k_logits.requires_grad = False
        self.algebra.params.requires_grad = True
        ema_beta = 0.95
        u_ema = None
        first_period_epoch = math.floor(epochs * logit_epoch_split) + 1
        for epoch in range(1, epochs + 1):
            if epoch == first_period_epoch:
                self.algebra.k_logits.requires_grad = True
                self.algebra.params.requires_grad = False
                if u_ema is not None:
                    with torch.no_grad():
                        self.algebra.params.copy_(u_ema / u_ema.norm(dim=1, keepdim=True))
            for x in self.loader():
                for i in range(self.n_gen):
                    # Gumbel softmax line used in experiments to imitate randomness in previous experiments
                    # _ = F.gumbel_softmax(self.algebra.k_logits[i], tau=tau, hard=False)
                    y = F.softmax(self.algebra.k_logits[i] / tau, dim=-1)
                    k_vals = self.algebra.k_space
                    angles = torch.tensor(list({float(2 * np.pi * j / k) for k in k_vals for j in range(1, k)}), device=self.device)
                    idx = torch.randint(angles.shape[0], (disc_samples,)).to(self.device)
                    thetas = angles[idx]
                    lie_basis = self.algebra.get_matrices()
                    G = [torch.matrix_exp((theta / self.algebra.params[i].detach().norm()) * lie_basis[i]) for theta in thetas]
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
                    loss_d = torch.stack([-torch.mean(torch.log(d_x + 1e-8) + torch.log(1 - self.d(g_x_i) + 1e-8)) for g_x_i in g_x]).mean()

                    # train discriminator
                    d_optim.zero_grad()
                    loss_d.backward()
                    d_optim.step()

                    # generator loss
                    g_optim.zero_grad()
                    if epoch > epochs * logit_epoch_split:
                        with torch.no_grad():
                            g_losses = []
                            for k in k_vals:
                                g_x = self.g(x, j=i, k=k, sampling_size=m_sample_size)
                                g_losses.append(-torch.stack([torch.log(self.d(g_x_i) + 1e-8) for g_x_i in g_x]).mean())
                        loss_g = (y * torch.stack(g_losses)).sum()
                    else:
                        g_losses = []
                        for k in k_vals:
                            g_x = self.g(x, j=i, k=k, sampling_size=m_sample_size)
                            g_losses.append(-torch.stack([torch.log(self.d(g_x_i) + 1e-8) for g_x_i in g_x]).mean())
                        loss_g = self.smoothmin(torch.stack(g_losses[1::]), rho=rho)

                    # k entropy regularisation
                    p = torch.softmax(self.algebra.k_logits[i], dim=0)
                    loss_e = lambda_e * (p * torch.log(p)).sum()

                    # orthogonality regularisation
                    loss_ortho = 0
                    if self.n_gen > 1:
                        loss_ortho = lambda_o * torch.stack([self.algebra.params[a] @ self.algebra.params[b].t() for a in range(self.n_gen) for b in range(a + 1, self.n_gen)]).pow(2).mean()

                    # train generator
                    loss_total = loss_g + loss_ortho + loss_e
                    loss_total.backward()
                    g_optim.step()
                    with torch.no_grad():
                        self.algebra.params /= self.algebra.params.norm(dim=1, keepdim=True)

            if epoch < first_period_epoch:
                with torch.no_grad():
                    u = self.algebra.params / self.algebra.params.norm(dim=1, keepdim=True)
                    u_ema = u.clone() if u_ema is None else ema_beta * u_ema + (1 - ema_beta) * u

            if print_losses:
                print(f"\nEpoch {epoch} | D loss: {loss_d:.4f} | " +
                      f"G loss: {loss_g:.4f} | ")
                print("Lie Algebra:")
                print(self.algebra.get_matrices())
                print("Weights:")
                print(torch.softmax(self.algebra.k_logits, dim=1).detach().round(decimals=3))
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
