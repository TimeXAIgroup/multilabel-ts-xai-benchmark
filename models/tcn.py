"""TCN torch implementation from https://github.com/timeseriesAI/tsai.

S. Bai, J. Z. Kolter, and V. Koltun,
“An Empirical Evaluation of Generic Convolutional and Recurrent Networks
for Sequence Modeling.” arXiv, 2018. doi: 10.48550/ARXIV.1803.01271.
"""

from tsai.models.TCN import TCN as TCN_tsai


class TCN(TCN_tsai):
    """TCN model from [1], pytorch implementation from https://github.com/timeseriesAI/tsai.

    [1] S. Bai, J. Z. Kolter, and V. Koltun,
    “An Empirical Evaluation of Generic Convolutional and Recurrent Networks
    for Sequence Modeling.” arXiv, 2018. doi: 10.48550/ARXIV.1803.01271."""
    def __init__(self,
                 in_channels,
                 num_classes,
                 **kwargs):
        # wrap for matching parameter names
        super(TCN, self).__init__(c_in=in_channels,
                                  c_out=num_classes,
                                  **kwargs)