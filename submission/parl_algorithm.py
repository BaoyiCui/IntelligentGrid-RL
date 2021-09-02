import parl
import paddle
from submission.parl_model import *

OBS_DIM = 620
ACT_DIM = 54

model = ParlModel(OBS_DIM, ACT_DIM)
algorithm = parl.algorithms.PolicyGradient(model, lr=1e-3)

