"""
Wrapped MOMENT Foundation Model from https://github.com/moment-timeseries-foundation-model/moment

M. Goswami, K. Szafer, A. Choudhry, Y. Cai, S. Li, and A. Dubrawski,
“MOMENT: A Family of Open Time-series Foundation Models,”
in Proceedings of the 41st International Conference on Machine Learning,
in ICML 24. Vienna, Austria: JMLR.org, 2024.
"""

from momentfm import MOMENTPipeline
from torch.nn import Module


class Moment(Module):
    """Pretrained MOMENT Foundation Model from [1], https://github.com/moment-timeseries-foundation-model/moment

    Note that training will only fine-tune the model's classification head and not the full model.

    [1] M. Goswami, K. Szafer, A. Choudhry, Y. Cai, S. Li, and A. Dubrawski,
    “MOMENT: A Family of Open Time-series Foundation Models,”
    in Proceedings of the 41st International Conference on Machine Learning,
    in ICML 24. Vienna, Austria: JMLR.org, 2024.
    """
    def __init__(self,
                 in_channels: int,
                 num_classes: int,
                 model_size: str = "base", # options: "small" | "base" | "large"
                 ):
        super(Moment, self).__init__()

        # load pretrained model
        self.model = MOMENTPipeline.from_pretrained(
            f"AutonLab/MOMENT-1-{model_size}",
            model_kwargs={
                'task_name': 'classification',
                'n_channels': in_channels,
                'num_class': num_classes
            },
        )
        self.model.init()

    def forward(self, x):
        return self.model.forward(x_enc=x).logits
