from sklearn.linear_model import RidgeClassifierCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sktime.transformations.rocket import MiniRocketMultivariate


class MiniRocket(Pipeline):
    """Implementation of MiniRocket classifier for multi-label classification. For further details on MiniRocket, see [1].

    [1] A. Dempster, D. F. Schmidt, and G. I. Webb. 2021.
    MiniRocket: A Very Fast (Almost) Deterministic Transform for Time Series Classification.
    In Proceedings of the 27th ACM SIGKDD Conference on Knowledge Discovery & Data Mining (KDD '21).
    Association for Computing Machinery, New York, NY, USA, 248–257. https://doi.org/10.1145/3447548.3467231
    """
    def __init__(self,):
        steps = [('minirocketmultivariate', MiniRocketMultivariate()),
                 ('scaler', StandardScaler()),
                 ('classifier', MultiOutputClassifier(RidgeClassifierCV()))]

        super(MiniRocket, self).__init__(steps=steps)
