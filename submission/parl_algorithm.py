import parl
import paddle
from submission.parl_model import *

OBS_DIM = 620
ACT_DIM = 54


ACTOR_LR = 1e-3  # Actor网络的 learning rate
CRITIC_LR = 1e-3  # Critic网络的 learning rate

GAMMA = 0.99      # reward 的衰减因子
TAU = 0.001       # 软更新的系数

model = ParlModel(OBS_DIM, ACT_DIM)
algorithm = parl.algorithms.DDPG(model, gamma=GAMMA, tau=TAU, actor_lr=ACTOR_LR, critic_lr=CRITIC_LR)

