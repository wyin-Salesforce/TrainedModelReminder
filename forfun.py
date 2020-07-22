import torch

a=torch.randn(5, 3)
print('original a:', a)
indices = torch.tensor([0,0,1,1,0])
changed_places = torch.nonzero(indices)
a[changed_places, 0] = 1 #- a[changed_places, 0]
print('new a:', a)
