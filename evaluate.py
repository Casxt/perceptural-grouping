import torch

if __name__ == '__main__':
    a = torch.tensor(3., requires_grad=True)
    b = torch.tensor(2., requires_grad=True)
    z = a * b
    z.backward()
    print(a.grad, b.grad) # tensor(2.) tensor(3.)
