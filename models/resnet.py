"""ResNet torch implementation from https://github.com/timeseriesAI/tsai.

Z. Wang, W. Yan, and T. Oates,
“Time series classification from scratch with deep neural networks: A strong baseline,”
in 2017 International Joint Conference on Neural Networks (IJCNN), IEEE, May 2017, pp. 1578–1585.
doi: 10.1109/ijcnn.2017.7966039.
"""

from tsai.models.ResNet import ResNet as ResNet_tsai


class ResNet(ResNet_tsai):
    """ResNet model from [1], pytorch implementation from https://github.com/timeseriesAI/tsai.

    [1] Z. Wang, W. Yan, and T. Oates,
    “Time series classification from scratch with deep neural networks: A strong baseline,”
    in 2017 International Joint Conference on Neural Networks (IJCNN), IEEE, May 2017, pp. 1578–1585.
    doi: 10.1109/ijcnn.2017.7966039."""
    def __init__(self,
                 in_channels: int,
                 num_classes: int,):
        # wrap for matching parameter names
        super(ResNet, self).__init__(c_in=in_channels,
                                     c_out=num_classes,)