import torch
import numpy as np


a=np.array([[[1,2,3],[4,5,6]]])
unpermuted=torch.tensor(a)
print(unpermuted)
print(unpermuted.size())  #  ——>  torch.Size([1, 2, 3])
permuted=unpermuted.permute(2,0,1)
print(permuted)
print(permuted.size())     #  ——>  torch.Size([3, 1, 2])