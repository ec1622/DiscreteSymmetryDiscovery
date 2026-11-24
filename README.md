# Discrete Rotational Symmetry Discovery in Point Clouds

Code for *A Generative Framework for Discrete Rotational Symmetry Discovery in Point Clouds*.

## Requirements

`torch`, `numpy`

## Usage

```python
import torch
from framework import Model

# (N, 2) or (N, 3) point cloud
X = ...
# normalise to [-1, 1]
X = X / X.abs().max()

model = Model(dim=3, n_gen=1)
model.insert_data(X, batch_size=3)
model.train()

# learned generator
L = model.algebra.get_matrices()[0]
# distribution over periods
q = torch.softmax(model.algebra.k_logits[0], dim=0)
```

For point clouds not centred on their symmetry axis, use `form="dse"` and set the centre to the mean before training:

```python
model = Model(dim=3, n_gen=1, form="dse")
model.set_centre(X.mean(dim=0))
```

Settings for each experiment are given in the appendix of the paper. See the docstrings of `Model` and `Model.train` for all options.
