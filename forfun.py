import torch

m = torch.nn.Sigmoid()
# loss = torch.nn.BCELoss()
# input = torch.randn(3, requires_grad=True)
# target = torch.empty(3).random_(2)
# output = loss(input, target)
print(m(0.0))
