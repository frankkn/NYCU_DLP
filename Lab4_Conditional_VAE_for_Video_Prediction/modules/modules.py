import torch.nn as nn
import torch
from .layers import DepthConvBlock, ResidualBlock
from torch.autograd import Variable


__all__ = [
    "Generator",
    "RGB_Encoder",
    "Gaussian_Predictor",
    "Decoder_Fusion",
    "Label_Encoder"
]

class Generator(nn.Sequential):
    '''
    將輸入的特徵逐步轉化為輸出圖像。Generator在VAE中用於生成圖像。
    '''
    def __init__(self, input_nc, output_nc):
        super(Generator, self).__init__(
            DepthConvBlock(input_nc, input_nc),
            ResidualBlock(input_nc, input_nc//2),
            DepthConvBlock(input_nc//2, input_nc//2),
            ResidualBlock(input_nc//2, input_nc//4),
            DepthConvBlock(input_nc//4, input_nc//4),
            ResidualBlock(input_nc//4, input_nc//8),
            DepthConvBlock(input_nc//8, input_nc//8),
            nn.Conv2d(input_nc//8, 3, 1)
        )
        
    def forward(self, input):
        # Decode the concatenated tensor to generate the output image
        return super().forward(input)
    
    
    
class RGB_Encoder(nn.Sequential):
    '''
    將輸入的RGB圖像轉化為特徵表示。
    '''
    def __init__(self, in_chans, out_chans):
        # channel數逐漸增加，用以蒐集更高級的特徵。
        super(RGB_Encoder, self).__init__(
            ResidualBlock(in_chans, out_chans//8),
            DepthConvBlock(out_chans//8, out_chans//8),
            ResidualBlock(out_chans//8, out_chans//4),
            DepthConvBlock(out_chans//4, out_chans//4),
            ResidualBlock(out_chans//4, out_chans//2),
            DepthConvBlock(out_chans//2, out_chans//2),
            nn.Conv2d(out_chans//2, out_chans, 3, padding=1),
        )  
        
    def forward(self, image):
        return super().forward(image)
    

    
    
    
class Label_Encoder(nn.Sequential):
    '''
    用於將輸入的label轉化為特徵表示。
    '''
    def __init__(self, in_chans, out_chans, norm_layer=nn.BatchNorm2d):
        super(Label_Encoder, self).__init__(
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_chans, out_chans//2, kernel_size=7, padding=0),
            norm_layer(out_chans//2),
            nn.LeakyReLU(True),
            ResidualBlock(in_ch=out_chans//2, out_ch=out_chans)
        )  
        
    def forward(self, image):
        return super().forward(image)
    
    
class Gaussian_Predictor(nn.Sequential):
    '''
    接收圖像和label的特徵，用於預測生成圖像分布的高斯參數(均值和變異數)。
    '''
    def __init__(self, in_chans=48, out_chans=96):
        super(Gaussian_Predictor, self).__init__(
            ResidualBlock(in_chans, out_chans//4),
            DepthConvBlock(out_chans//4, out_chans//4),
            ResidualBlock(out_chans//4, out_chans//2),
            DepthConvBlock(out_chans//2, out_chans//2),
            ResidualBlock(out_chans//2, out_chans),
            nn.LeakyReLU(True),
            nn.Conv2d(out_chans, out_chans*2, kernel_size=1)
        )
        
    def reparameterize(self, mu, logvar):
        '''
        從均值(mu)和變異數的對數(logvar)中採樣生成潛在變量，以便進行反向傳播和優化。
        '''
        # Sample from the posterior distribution using the reparameterization trick
        std = torch.exp(logvar/2)
        eps = torch.randn_like(std) # return a same size tensor as std, sample from Normal distribution
        return mu + eps * std

    def forward(self, img, label):
        # Predict the posterior distribution parameters
        feature = torch.cat([img, label], dim=1)
        parm = super().forward(feature)
        mu, logvar = torch.chunk(parm, 2, dim=1)
        # z是模型使用reparameterization trick從給定的均值(mu)和對數變異數(logvar)中生成的潛在變數。
        # 這個潛在變數通常被視為模型在隱藏空間中的表示。
        # VAE中，z可以看作是從隱藏空間中隨機生成的一個點，用於生成數據。
        z = self.reparameterize(mu, logvar) 

        return z, mu, logvar
    
    
class Decoder_Fusion(nn.Sequential):
    '''
    接收圖像、label和Gaussian_Predictor的參數，用於生成最終的圖像。
    '''
    def __init__(self, in_chans=48, out_chans=96):
        super().__init__(
            DepthConvBlock(in_chans, in_chans),
            ResidualBlock(in_chans, in_chans//4),
            DepthConvBlock(in_chans//4, in_chans//2),
            ResidualBlock(in_chans//2, in_chans//2),
            DepthConvBlock(in_chans//2, out_chans//2),
            nn.Conv2d(out_chans//2, out_chans, 1, 1)
        )
        
    def forward(self, img, label, parm):
        # Concatenate the sampled noise vector with feature representations
        # print("img size:", img.size())
        # print("label size:", label.size())
        # print("parm size:", parm.size())

        feature = torch.cat([img, label, parm], dim=1)
        return super().forward(feature)
    

    
        
    
if __name__ == '__main__':
    pass
