import torch

y = 4

test = torch.zeros((2,3) + (y,))

print("Tensor shape:", test.shape)
print("Tensor values:\n", test)

x = ([1,3,4,5],[3,5,3,3],[1,7,5,5])

test[0] = torch.Tensor(x)

print("Tensor values:\n", test)