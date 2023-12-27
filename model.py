import torch
import torch.nn as nn

class conv_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_c)

        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_c)

        self.relu = nn.ReLU()

    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        return x

class encoder_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.conv = conv_block(in_c, out_c)
        self.pool = nn.MaxPool2d((2, 2))

    def forward(self, inputs):
        x = self.conv(inputs)
        p = self.pool(x)

        return x, p

class SE(nn.Module):
    def __init__(self, in_channel, ratio=4):
        super().__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=1)

        self.fc1 = nn.Linear(in_features=in_channel, out_features=in_channel//ratio, bias=False)

        self.relu = nn.ReLU()
 
        self.fc2 = nn.Linear(in_features=in_channel//ratio, out_features=in_channel, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):  
 
        b, c, h, w = inputs.shape

        x = self.avg_pool(inputs)
 
        x = x.view([b,c])
        
        x = self.fc1(x)
 
        x = self.relu(x)
 
        x = self.fc2(x)
 
        x = self.sigmoid(x)
        
        x = x.view([b,c,1,1])
        
        outputs = x * inputs

        return outputs

class APF(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv = nn.Conv2d(in_c//2, out_c, kernel_size=3, padding=1)
        self.maxpool = nn.MaxPool2d((2, 2))
        self.avgpool = nn.AvgPool2d((2, 2))
        self.convlast = nn.Conv2d(in_c, out_c, kernel_size=1, padding=0)
        self.se = SE(in_c//2)

    def forward(self, inputs):
        inputs = self.se(inputs)
        x1 = self.upsample(inputs)
        x2 = self.conv(x1)
        p = 0.5*self.maxpool(x2)+0.5*self.avgpool(x2)
        out = torch.cat([inputs,p], dim=1)
        out1 = self.convlast(out)

        return out1

class decoder_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.up = nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2, padding=0)
        self.conv = conv_block(out_c+out_c, out_c)
        self.sp = sp(in_c, out_c)

    def forward(self, inputs, skip):
        x = self.up(inputs)
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return x
    
class decoder_block_apf(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.up = nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2, padding=0)
        self.conv = conv_block(out_c+out_c, out_c)
        self.apf = APF(in_c, out_c)

    def forward(self, inputs, skip):
        x = self.up(inputs)
        skip = self.apf(skip)
        x = torch.cat([x, skip], axis=1)
        x = self.conv(x)
        return x

class build_unet(nn.Module):
    def __init__(self):
        super().__init__()

        """ Encoder """
        self.e1 = encoder_block(3, 64)
        self.e2 = encoder_block(64, 128)
        self.e3 = encoder_block(128, 256)
        self.e4 = encoder_block(256, 512)

        """ Bottleneck """
        self.b = conv_block(512, 1024)

        """ Decoder """
        self.d1 = decoder_block(1024, 512)
        self.d2 = decoder_block(512, 256)
        self.d3 = decoder_block_apf(256, 128)
        self.d4 = decoder_block(128, 64)

        """ Classifier """
        self.outputs = nn.Conv2d(64, 1, kernel_size=1, padding=0)

        self.conv1 = nn.Conv2d(512, 1024, kernel_size=1, padding=0)
        self.conv2 = nn.Conv2d(256, 512, kernel_size=1, padding=0)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=1, padding=0)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=1, padding=0)

        self.up1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2, padding=0)
        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2, padding=0)
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2, padding=0)
        self.up4 = nn.ConvTranspose2d(128, 1, kernel_size=2, stride=2, padding=0)

    def forward(self, inputs):
        """ Encoder """
        s1, p1 = self.e1(inputs)
        s2, p2 = self.e2(p1)
        s3, p3 = self.e3(p2)
        s4, p4 = self.e4(p3)

        """ Bottleneck """
        b = self.b(p4)

        """ Decoder """
        d1 = self.d1(b, s4)
        d2 = self.d2(d1, s3)
        d3 = self.d3(d2, s2)
        d4 = self.d4(d3, s1)

        x1 = self.conv1(p4)
        y1 = torch.add(b,x1)
        y11 = self.up1(y1)
        x2 = self.conv2(p3)
        y2 = torch.add(d1,x2)
        y22 = 0.3*y11+0.7*y2
        y22 = self.up2(y22)
        x3 = self.conv3(p2)
        y3 = torch.add(d2,x3)
        y33 = 0.3*y22+0.7*y3
        y33 = self.up3(y33)
        x4 = self.conv4(p1)
        y4 = torch.add(d3,x4)
        y44 = 0.3*y33+0.7*y4
        y44 = self.up4(y44)

        outputs = self.outputs(d4)

        outputs = 0.3*y44+0.7*outputs

        outputs = torch.sigmoid(outputs)

        return outputs
    
if __name__ == "__main__":
    x = torch.randn(([2, 3, 512, 512]))
    f = build_unet()
    y = f(x)
    print(y.shape)