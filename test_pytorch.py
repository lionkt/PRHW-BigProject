import torch
import matplotlib.pyplot as plt

N, D_in, H, D_out = 64, 1000, 100, 10

device = torch.device("cuda:1")
# device = torch.device("cpu")

x = torch.rand(N, D_in, device=device)
y = torch.rand(N, D_out, device=device)

def default_NN():
    # 利用torch.nn.Sequential搭建的级联网络
    model = torch.nn.Sequential(
        torch.nn.Linear(D_in, H),
        torch.nn.ReLU(),
        torch.nn.Linear(H, D_out), )

    if device.type != 'cpu':
        model = model.cuda(device=device)
    loss_func = torch.nn.MSELoss(size_average=False)

    lr = 1e-4
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for step in range(5000):
        y_pred = model(x)
        loss = loss_func(y_pred, y)
        print(step, loss.data.cpu().numpy())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return model




def selfBuild_NN():
    # 通过自建class构建神经网络
    class TwoLayerNet(torch.nn.Module):
        def  __init__(self,D_in, H, D_out):
            super(TwoLayerNet, self).__init__()
            self.linear1 = torch.nn.Linear(D_in, H)
            self.linear2 = torch.nn.Linear(H, D_out)

        def forward(self, x):
            h_relu = self.linear1(x).clamp(min=0)
            relu_func = torch.nn.ReLU()
            y_pred = self.linear2(relu_func(h_relu))
            return y_pred

    # instance self-build nn
    model = TwoLayerNet(D_in, H, D_out)
    if device.type != 'cpu':
        model = model.cuda(device=device)
    loss_func = torch.nn.MSELoss(size_average=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)

    for step in range(5000):
        y_pred = model(x)
        loss = loss_func(y_pred, y)
        print(step, loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return model






if __name__ == '__main__':
    selfBuild_NN()
