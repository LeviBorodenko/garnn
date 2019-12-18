# GARNN [TensorFlow]
TensorFlow implementation of _Graphical Attention Recurrent Networks_ based on work by [Cirstea et al., 2019](https://milets19.github.io/papers/milets19_paper_8.pdf).

Moreover, we offer stand-alone implementations of the _Graph Attention Mechanism_ [(Veličković et al., 2017)](https://arxiv.org/abs/1710.10903) and _Diffusional Graph Convolution_ [(Li et al., 2017)](https://arxiv.org/pdf/1707.01926.pdf).

### Installation
Simply run `pip install garnn`. Dependencies are `numpy; tensorflow`.

### Features

The core data structure is the _graph signal_. If we have N nodes in a graph each having F observed features then the graph signal is the tensor with shape (batch, N, F) corresponding to the data produced by all nodes. Often we have sequences of graph signals in a time series. We will call them _temporal_ graph signals and assume a shape of (batch, time steps, N, F). We also need to know the adjacency matrix E of the underlying graph with shape (N, N).

#### Non-Temporal Data (batch, N, F)
All but the recurrent layers work with non - temporal data, i.e. the data points are individual graph signals and not sequences of graph signals.

The `AttentionMechanism` found in `garnn.components.attention` will take a graph signal and return an attention matrix as described in [Veličković et al., 2017](https://arxiv.org/abs/1710.10903).

The layer is initiated with the following parameters:

| Parameter | Function |
|:------------- | :--------|
|`F` (required) | Dimension of internal embedding.|
|`adjacency_matrix` (required) | Adjacency matrix of the graph.|
|`num_heads` (default: 1) | Number of attention matrices that are averaged to return the output attention.|
|`use_reverse_diffusion` (default: True) | Whether or not to calculate A_in and A_out as done by [Cirstea et al., 2019](https://milets19.github.io/papers/milets19_paper_8.pdf). If E is symmetric then the value will be set to False.|

The output is of shape (batch, N, N). If `use_reverse_diffusion` is true then we obtain 2 attention matrices and thus the shape is (batch, 2, N, N).

The `GraphDiffusionConvolution` layer in `garnn.layers.diffconv` offers diffusion graph convolution as described by [(Li et al., 2017)](https://arxiv.org/pdf/1707.01926.pdf). It operates on a tuple containing a graph signal X and an adjacency matrix A (usually an attention matrix returned by an attention mechanism) and is initiated with the following parameters

| Parameter | Function |
|:------------- | :--------|
|`features` (required) | Number output features. Q in the paper.|
|`num_diffusion_steps` (required) | Number of hops done by the diffusion process. K in the paper. |

There are more specialised parameters like regularisers and initialisers -- those can be found in the doc string. The convolutional layer returns a diffused graph signal of shape (batch, N, Q).

Thus, if we have 10 nodes with 5 features each and we would like to apply diffusion graph convolution with 20 features using a 5-head attention mechanism with an internal embedding of 64 units then we would need to run

```python
from garnn.components.attention import AttentionMechanism
from garnn.layers.diffconv import GraphDiffusionConvolution
from tensorflow.keras.layers import Input

# input of 10 by 5 graph signal
inputs = Input(shape=(10, 5))

# Initiating attention mechanism. Make sure you define E
Attn = AttentionMechanism(64, adjacency_matrix=E, num_heads=5)(inputs)

# Now the convolutional layer. Make sure you use the correct order in the
# input-tuple: Graph signal is always first!
output = GraphDiffusionConvolution(
    features=10, num_diffusion_steps=5)((inputs, Attn))
```


#### Temporal Data (batch, time steps, N, F)

Both `AttentionMechanism` and `DiffusionGraphConvolution` naturally extend to temporal graph signals. The output now simply has an additional time steps dimension.

The `garnn_gru` layer found in `garnn.components.garnn_gru` is the diffusional & attention-based GRU introduced by [Cirstea et al., 2019](https://milets19.github.io/papers/milets19_paper_8.pdf). It operates on temporal graph signals and an attention mechanism. Initiate with

| Parameter | Function |
|:------------- | :--------|
|`num_hidden_features` (required) | Number of features in the hidden state. (Q in the paper)|
|`num_diffusion_steps` (default: 5) | Number of hops done by the diffusion process in all internal convolutions. K in the paper.|
|`return_sequence` (default: False) | Whether or not the RNN should return the hidden state at each time step or only at the final time step. Set it to `True` if you stack another RNN layer on top of this one.|

Hence, if we would like to rebuild a model similar to the one used by [Cirstea et al., 2019](https://milets19.github.io/papers/milets19_paper_8.pdf) then one needs to run

```python
import numpy as np
from tensorflow.keras import Model
from tensorflow.keras.layers import Input

from garnn.components.attention import AttentionMechanism
from garnn.components.garnn_gru import garnn_gru

# We assume we have 207 nodes with 10 features each.

# For simplicity we create a random adjacency matrix E
E = np.random.randint(0, 2, size=(207, 207))

# Input of the temporal graph signals. Note that "None" allows
# us to pass in variable length time series.
X = Input(shape=(None, 207, 10))

# creating attention mechanism with 3 heads and an
# embedding size of 16
A = AttentionMechanism(16, adjacency_matrix=E, num_heads=3)(X)

# now piping X and A into the 2 GRU layers
# First layer streches the features into 64 diffused features.
# We assume the we are using 6 hop diffusions.
gru_1 = garnn_gru(num_hidden_features=64, num_diffusion_steps=6, return_sequences=True)(
    (X, A)
)

# And then we use another GRU to shrink it back to 1 feature - the feature
# that we are predicting. We use the same attention for this layer, but note that
# we could also introduce a new attention mechanism for the next GRU.
output = garnn_gru(num_hidden_features=1, num_diffusion_steps=6)((gru_1, A))

garnn_model = Model(inputs=X, outputs=output)
```

### Contribute
Bug reports, fixes and additional features are always welcome! Make sure to run the tests with `python setup.py test` and write your own for new features. Thanks.
