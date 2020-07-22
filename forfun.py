import torch

a=torch.randn(5, 3)
print('original a:', a)
indices = torch.tensor([0,0,1,1,0])
a[torch.nonzero(indices), 0] = 1 - a[torch.nonzero(indices), 0]
print('new a:', a)
