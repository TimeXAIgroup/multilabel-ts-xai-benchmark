from types import SimpleNamespace

import torch.nn as nn


class _CNN_Turbe(nn.Module):
    """ CNN  implementation from https://github.com/hturbe/InterpretTime

    Turbé, H., Bjelogrlic, M., Lovis, C. et al.
    Evaluation of post-hoc interpretability methods in time-series classification.
    Nat Mach Intell 5, 250–260 (2023). https://doi.org/10.1038/s42256-023-00620-w
    """
    def __init__(self, hparams, **kwargs):
        super().__init__()

        self.length_sequence = kwargs["length_sequence"]
        self.n_features = kwargs["n_features"]
        self.n_classes = kwargs["n_classes"]

        self.hparams = SimpleNamespace(
            num_classes=kwargs['n_classes'],
            filter_arrays=hparams["cell_array"],
            dropout=hparams.get("dropout", 0.0),
            kernel_size=hparams["kernel_size"],
            act_fn_name=hparams['act_fn_name'],
            act_fn=hparams['act_fn'],
        )

        self._create_network()

    def _create_network(self):
        blocks = []
        kernel_size = self.hparams.kernel_size
        dropout = self.hparams.dropout
        for idx, filters in enumerate(self.hparams.filter_arrays):
            if idx == 0:
                blocks.append(nn.Sequential(
                    nn.Conv1d(
                        in_channels=self.n_features,
                        out_channels=filters,
                        kernel_size=kernel_size,
                        stride=1,
                    ), self.hparams.act_fn(),
                    nn.MaxPool1d(kernel_size=2),
                    nn.Dropout(p=dropout)
                ))

            else:
                blocks.append(nn.Sequential(
                    nn.Conv1d(
                        in_channels=self.hparams.filter_arrays[idx - 1],
                        out_channels=filters,
                        kernel_size=kernel_size,
                        stride=1,
                    ), self.hparams.act_fn(),
                    nn.MaxPool1d(kernel_size=2),
                    nn.Dropout(p=dropout)
                ))

        self.blocks = nn.Sequential(*blocks)

        self.output_net = nn.Sequential(
            nn.AdaptiveAvgPool1d((1)), nn.Flatten(), nn.Linear(self.hparams.filter_arrays[-1],
                                                               self.hparams.num_classes)
        )

    def forward(self, x):
        x = self.blocks(x)
        x = self.output_net(x)

        return x


class CNN(_CNN_Turbe):
    """Wraps CNN  implementation from Turbé et al. [1]

    [1] Turbé, H., Bjelogrlic, M., Lovis, C. et al.
    Evaluation of post-hoc interpretability methods in time-series classification.
    Nat Mach Intell 5, 250–260 (2023). https://doi.org/10.1038/s42256-023-00620-w
    """

    def __init__(self,
                 in_channels,
                 num_classes,
                 **kwargs):

        # Set parameters to those of the CNN for synthetic data in [1]
        hparams = {"cell_array": [256, 256, 256],
                   "dropout": 0.3,
                   "kernel_size": 11,
                   "act_fn_name": "relu",
                   "act_fn": nn.ReLU,}

        kwargs.update({"length_sequence": 500,
                       "n_features": in_channels,
                       "n_classes": num_classes})

        super(CNN, self).__init__(hparams=hparams, **kwargs)