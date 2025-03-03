import torch
import torch.nn as nn
from torch.nn import Module
from torch import Tensor
import torch.optim as optim
import onnx
import os

class Test(Module):
    def __init__(self, size: int, *args, **kwargs):
        super().__init__()
        self.ls = nn.Sequential(
            *[
                nn.Sequential(
                    nn.Linear(size,size),
                    nn.LeakyReLU()
                )
                for _ in range(3)
            ]
        )

    def forward(self, x: Tensor):
        x = self.ls.forward(x)
        return torch.sigmoid(x)

model = Test(size=28*28)
opt = optim.Adam(params=model.parameters(),lr=0.01)
crit = nn.BCELoss()

for i in range(100):
    data = torch.rand((28,28)).reshape(1,-1)
    data = (data > 0.5).to(torch.float32)
    # print(data)
    target = torch.ones_like(data)
    pred = (torch.logical_xor(data,target)).to(torch.float32)
    opt.zero_grad()
    # print(pred)
    res = model.forward(data)
    # print(res)
    loss = crit.forward(res,pred)
    print(loss)
    loss.backward()
    opt.step()

torch.onnx.export(
    model=model,
    args=torch.randn(1,28*28),
    f=f'{os.path.dirname(__file__)}/model.onnx',
    export_params=True,
    input_names=['input'],
    output_names=['output'],
)