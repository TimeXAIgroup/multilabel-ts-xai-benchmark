"""
InceptionTime torch implementation from https://github.com/timeseriesAI/tsai/blob/main/tsai/models/InceptionTime.py

Fawaz, H. I., Lucas, B., Forestier, G., Pelletier, C., Schmidt, D. F., Weber, J., ... & Petitjean, F. (2019).
InceptionTime: Finding AlexNet for Time Series Classification. arXiv preprint arXiv:1909.04939.
Official InceptionTime tensorflow implementation: https://github.com/hfawaz/InceptionTime
"""
from tsai.models.InceptionTime import InceptionTime as InceptionTime_tsai


class InceptionTime(InceptionTime_tsai):
    """InceptionTime model from [1], pytorch implementation from https://github.com/timeseriesAI/tsai.

    [1] Fawaz, H. I., Lucas, B., Forestier, G., Pelletier, C., Schmidt, D. F., Weber, J., ... & Petitjean, F. (2019).
    InceptionTime: Finding AlexNet for Time Series Classification. arXiv preprint arXiv:1909.04939.
    Official InceptionTime tensorflow implementation: https://github.com/hfawaz/InceptionTime
    """
    def __init__(self,
                 in_channels,
                 num_classes,
                 **kwargs):
        # wrap for matching parameter names
        super(InceptionTime, self).__init__(c_in=in_channels,
                                            c_out=num_classes,
                                            **kwargs)
