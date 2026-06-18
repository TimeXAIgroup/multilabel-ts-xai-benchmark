"""
InceptionTime torch implementation from https://github.com/timeseriesAI/tsai

H. Ismail Fawaz et al., “InceptionTime: Finding AlexNet for time series classification,”
Data Mining and Knowledge Discovery, vol. 34, no. 6, pp. 1936–1962, Sep. 2020,
doi: 10.1007/s10618-020-00710-y.
Official InceptionTime tensorflow implementation: https://github.com/hfawaz/InceptionTime
"""
from tsai.models.InceptionTime import InceptionTime as InceptionTime_tsai


class InceptionTime(InceptionTime_tsai):
    """InceptionTime model from [1], pytorch implementation from https://github.com/timeseriesAI/tsai.

    [1] H. Ismail Fawaz et al., “InceptionTime: Finding AlexNet for time series classification,”
    Data Mining and Knowledge Discovery, vol. 34, no. 6, pp. 1936–1962, Sep. 2020,
    doi: 10.1007/s10618-020-00710-y.
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
