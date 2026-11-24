# Discrete Rotational Symmetry Discovery in Point Clouds

Code for *A Generative Framework for Discrete Rotational Symmetry Discovery in Point Clouds*.

## Requirements

`torch`, `numpy`

## Usage

```python
import torch
from framework import Model

X = ...                      # (N, 2) or (N, 3) point cloud
X = X / X.abs().max()        # normalise to [-1, 1]

model = Model(dim=3, n_gen=1, seed=1, device="cuda")
model.insert_data(X, batch_size=2)
model.train()

L = model.algebra.get_matrices()[0]                     # learned generator
q = torch.softmax(model.algebra.k_logits[0], dim=0)     # distribution over periods
print(model.algebra.k_space, q)
```

For point clouds not centred on their symmetry axis, use `form="dse"` and set the centre to the mean before training:

```python
model = Model(dim=3, n_gen=1, form="dse", seed=1, device="cuda")
model.set_centre(X.mean(dim=0))
```

Default settings are those used in the paper; see the docstrings of `Model` and `Model.train` for all options.
