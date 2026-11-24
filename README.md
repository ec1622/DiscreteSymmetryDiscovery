# Discrete Rotational Symmetry Discovery in Point Clouds

Code for *A Generative Framework for Discrete Rotational Symmetry Discovery in Point Clouds*.

The framework learns a rotation axis and a distribution over a candidate set of rotational periods, from which the cyclic symmetry group about that axis is read off directly. Training has two stages: an adversarial axis stage using a smooth minimum over the period losses, and a period stage that learns a categorical distribution over the candidate periods with entropy regularisation and a temperature-scaled softmax.

## Requirements

```
pip install -r requirements.txt
```

`torch`, `numpy`, and `biopython` (for the protein experiments only).

## Usage

All settings default to those used for the 3D synthetic shapes (Table 6 of the paper). The other experiments differ only in the arguments shown below.

Point clouds should be normalised to $[-1, 1]$ by dividing every coordinate by the maximum absolute coordinate.

### 3D synthetic shapes

```python
import torch
from framework import Model

X = ...                      # (N, 3) tensor, here N = 100,000
X = X / X.abs().max()

model = Model(dim=3, n_gen=1, seed=1, device="cuda")
model.insert_data(X, batch_size=2)
model.train()                # 200 epochs: 100 axis, 100 period
```

### 2D synthetic shapes

In 2D there is only one rotation plane, so the axis stage is reduced to a single epoch.

```python
model = Model(dim=2, n_gen=1, seed=1, device="cuda")
model.insert_data(X, batch_size=2)                # (N, 2) tensor, here N = 20,000
model.train(logit_epoch_split=0.005)              # 1 axis epoch, 199 period epochs
```

### Protein structures

Proteins use the off-centre parametrisation, with the centre set to the mean of the structure.

```python
from Bio.PDB import MMCIFParser

structure = MMCIFParser(QUIET=True).get_structure("protein", "7PBJ.cif")
X = torch.tensor([atom.get_coord() for atom in structure.get_atoms()])
X = X / X.abs().max()

model = Model(dim=3, n_gen=1, form="dse", seed=1, device="cuda")
model.set_centre(X.mean(dim=0))
model.insert_data(X, batch_size=3)
model.train(epochs=400)       # 200 axis, 200 period
```

All experiments in the paper use seeds 1 to 10.

## Reading the results

```python
L = model.algebra.get_matrices()[0]                        # learned generator
q = torch.softmax(model.algebra.k_logits[0], dim=0)        # period distribution
periods = model.algebra.k_space                            # [1, 2, ..., 10]
```

For 3D data, the learned axis is the null vector of the rotational part of `L` (the top-left 3x3 block when `form="dse"`). A run is non-trivial if $q_1 < 0.9$; the recovered cyclic group about the axis is $C_n$, where $n$ is the largest period with $q_n \ge 0.1$.

## Main arguments

| Argument | Default | Description |
|---|---|---|
| `k_vals` | `"1:10"` | Candidate period set |
| `rho` | `3.0` | Smooth minimum temperature |
| `tau` | `2.0` | Softmax temperature |
| `lambda_e` | `0.05` | Entropy coefficient |
| `glr`, `dlr`, `llr` | `4e-2`, `1e-3`, `2e-2` | Generator, discriminator and logit learning rates |
| `logit_epoch_split` | `0.5` | Fraction of epochs spent on the axis stage |
| `m_sample_size`, `disc_samples` | `6`, `6` | Angles sampled per period loss and per discriminator update |

Setting `cloud_data=False` replaces the PointNet discriminator with the per-point MLP used in the discriminator comparison, trained on the full point cloud at each step with `model.insert_data(X, batch=False)`.
