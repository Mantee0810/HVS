
import torch
import torch.nn as nn

class CBR(nn.Module):
    def __init__(self,in_channel,out_channel,ks,stride,padding,dilation=1,bias=False):
        super(CBR, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channel,out_channels=out_channel,kernel_size=ks,stride=stride,padding=padding,dilation=dilation,bias=bias)
        self.bn = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(inplace=True)

    def forward(self,x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class HVSModule(nn.Module):
    def __init__(self,device='cuda:1'):
        super(HVSModule, self).__init__()
        self.conv_v = None
        self.conv_h = None
        self.bn = None
        self.sig = nn.Sigmoid()
        self.device = device


    def forward(self,x):
        b,c,h,w = x.shape

        if self.conv_v == None:
            self.conv_v = CBR(in_channel=c,out_channel=c,ks=(1,w),stride=1,padding=0).to(self.device)
        if self.conv_h == None:
            self.conv_h = CBR(in_channel=c, out_channel=c, ks=(h,1), stride=1, padding=0).to(self.device)
        if self.bn == None:
            self.bn = nn.BatchNorm2d(c).to(self.device)
        if self.sig == None:
            self.sig = nn.Sigmoid().to(self.device)

        x_v = (self.conv_v(x)).repeat(1, 1, 1, w)
        x_h = (self.conv_h(x)).repeat(1, 1, h, 1)

        x_v = self.sig(x_v) * x
        x_h = self.sig(x_h) * x

        x = x_v + x_h + x

        return x


if __name__ == '__main__':
    input1 = torch.randn(size=(2,3,7,7))
    h = HVSModule()
    out = h(input1)
    # dbndc = DBNDC(3)
    # out = dbndc(input1)
    print(out.shape)
