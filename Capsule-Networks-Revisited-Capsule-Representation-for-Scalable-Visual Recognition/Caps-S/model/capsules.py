import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import pdb
import numba
from torch.autograd.variable import Variable
from torch.nn.modules.loss import _Loss

class BasicConv2d_activation(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d_activation, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=1, bias=False)
        # self.conv2 = nn.Conv2d(in_planes, out_planes,
        #                       kernel_size=kernel_size, stride=stride,
        #                       padding=padding, dilation=3, bias=False)
        # self.conv3 = nn.Conv2d(in_planes, out_planes,
        #                       kernel_size=kernel_size, stride=stride,
        #                       padding=padding, dilation=5, bias=False)
        # self.conv4 = nn.Conv2d(in_planes, out_planes,
        #                       kernel_size=kernel_size, stride=stride,
        #                       padding=padding, dilation=7, bias=False)       
        # self.conv_cat = BasicConv2d(4*out_planes, out_planes, 3, padding=1)
        self.bn = nn.BatchNorm2d(out_planes)
        #self.relu = nn.ReLU(inplace=False)
        self.sigmoid = nn.Sigmoid()
        # self.convd = RF(in_planes, out_planes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn(x)
        #x = self.convd(x)
        #x2 = self.conv2(x)
        #x2 = self.bn(x2)
        #x3 = self.conv3(x)
        #x3 = self.bn(x3)
        #x4 = self.conv4(x)
        #x4 = self.bn(x4)  
        #x = self.conv_cat(torch.cat((x1, x2, x3, x4), dim=1))
        x = self.sigmoid(x)
        return x
    
class BasicConv2d_activationRL(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d_activationRL, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=1, bias=False)
        # self.conv2 = nn.Conv2d(in_planes, out_planes,
        #                       kernel_size=kernel_size, stride=stride,
        #                       padding=padding, dilation=3, bias=False)
        # self.conv3 = nn.Conv2d(in_planes, out_planes,
        #                       kernel_size=kernel_size, stride=stride,
        #                       padding=padding, dilation=5, bias=False)
        # self.conv4 = nn.Conv2d(in_planes, out_planes,
        #                       kernel_size=kernel_size, stride=stride,
        #                       padding=padding, dilation=7, bias=False)       
        # self.conv_cat = BasicConv2d(4*out_planes, out_planes, 3, padding=1)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=False)
        self.sigmoid = nn.Sigmoid()
        # self.convd = RF(in_planes, out_planes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn(x)
        #x = self.convd(x)
        #x2 = self.conv2(x)
        #x2 = self.bn(x2)
        #x3 = self.conv3(x)
        #x3 = self.bn(x3)
        #x4 = self.conv4(x)
        #x4 = self.bn(x4)  
        #x = self.conv_cat(torch.cat((x1, x2, x3, x4), dim=1))
        x = self.relu(x)
        return x


 
class PrimaryCaps(nn.Module):
    r"""Creates a primary convolutional capsule layer
    that outputs a pose matrix and an activation.

    Note that for computation convenience, pose matrix
    are stored in first part while the activations are
    stored in the second part.

    Args:
        A: output of the normal conv layer
        B: number of types of capsules
        K: kernel size of convolution
        P: size of pose matrix is P*P
        stride: stride of convolution

    Shape:
        input:  (*, A, h, w)
        output: (*, h', w', B*(P*P+1))
        h', w' is computed the same way as convolution layer
        parameter size is: K*K*A*B*P*P + B*P*P
    """
    def __init__(self, A=32, B=32, K=1, P=4, stride=1):
        super(PrimaryCaps, self).__init__()
        self.pose = nn.Conv2d(in_channels=A, out_channels=B*P*P,
                            kernel_size=K, stride=stride, bias=True)
        self.a = nn.Conv2d(in_channels=A, out_channels=B,
                            kernel_size=K, stride=stride, bias=True)
        self.sigmoid = nn.Sigmoid()
        self.B = B
        self.P = P
    def forward(self, x):
        p = self.pose(x)
        a = self.a(x)
        a = self.sigmoid(a)
        p = p*a.repeat(1,self.P*self.P,1,1)
        
        out = torch.cat([p, a], dim=1)   #[b, B*(16+1), 14, 14]
        # out = out.permute(0, 2, 3, 1)
        return out
    
class PrimaryCapsClass(nn.Module):
    r"""Creates a primary convolutional capsule layer
    that outputs a pose matrix and an activation.

    Note that for computation convenience, pose matrix
    are stored in first part while the activations are
    stored in the second part.

    Args:
        A: output of the normal conv layer
        B: number of types of capsules
        K: kernel size of convolution
        P: size of pose matrix is P*P
        stride: stride of convolution

    Shape:
        input:  (*, A, h, w)
        output: (*, h', w', B*(P*P+1))
        h', w' is computed the same way as convolution layer
        parameter size is: K*K*A*B*P*P + B*P*P
    """
    def __init__(self, A=32, B=32, K=1, P=4, stride=1, h=2, w=2):
        super(PrimaryCapsClass, self).__init__()
        self.pose = nn.Conv2d(in_channels=A*h*w, out_channels=B*P*P,
                            kernel_size=K, stride=stride, bias=True)
        self.a = nn.Conv2d(in_channels=A*h*w, out_channels=B,
                            kernel_size=K, stride=stride, bias=True)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        b, T, h, w = x.shape
        x = x.reshape(b, h * w * T, 1, 1)
        p = self.pose(x)
        a = self.a(x)
        a = self.sigmoid(a)
        out = torch.cat([p, a], dim=1)   #[b, B*(16+1), 14, 14]
        # out = out.permute(0, 2, 3, 1)
        return out
    


class GroupBatchnorm2d(nn.Module):
    def __init__(self, c_num:int, 
                 group_num:int = 16, 
                 eps:float = 1e-10
                 ):
        super(GroupBatchnorm2d,self).__init__()
        assert c_num    >= group_num
        self.group_num  = group_num
        self.gamma      = nn.Parameter( torch.randn(c_num, 1, 1)    )
        self.beta       = nn.Parameter( torch.zeros(c_num, 1, 1)    )
        self.eps        = eps

    def forward(self, x):
        N, C, H, W  = x.size()
        x           = x.view(   N, self.group_num, -1   )
        mean        = x.mean(   dim = 2, keepdim = True )
        std         = x.std (   dim = 2, keepdim = True )
        x           = (x - mean) / (std+self.eps)
        x           = x.view(N, C, H, W)
        return x * self.gamma + self.beta

class ConvCaps(nn.Module):
    r"""Create a convolutional capsule layer
    that transfer capsule layer L to capsule layer L+1
    by EM routing.

    Args:
        B: input number of types of capsules
        C: output number on types of capsules
        K: kernel size of convolution
        P: size of pose matrix is P*P
        stride: stride of convolution
        iters: number of EM iterations
        coor_add: use scaled coordinate addition or not
        w_shared: share transformation matrix across w*h.

    Shape:
        input:  (*, h,  w, B*(P*P+1))
        output: (*, h', w', C*(P*P+1))
        h', w' is computed the same way as convolution layer
        parameter size is: K*K*B*C*P*P + B*P*P
    """
    def __init__(self, B=32, C=32, K=3, P=4, stride=2, iters=3, channel1 = 272,
                 coor_add=False, w_shared=False):
        super(ConvCaps, self).__init__()
        # TODO: lambda scheduler
        # Note that .contiguous() for 3+ dimensional tensors is very slow
        self.B = B
        self.C = C
        self.K = K
        self.P = P
        self.psize = P*P
        self.stride = stride
        self.iters = iters
        self.coor_add = coor_add
        self.w_shared = w_shared
        # constant
        self.eps = 1e-8
        # self._lambda = 1e-03
        # self.ln_2pi = torch.cuda.FloatTensor(1).fill_(math.log(2*math.pi))
        # params
        # Note that \beta_u and \beta_a are per capsule type,
        # which are stated at https://openreview.net/forum?id=HJWLfGWRb&noteId=rJUY2VdbM
        # self.beta_u = nn.Parameter(torch.zeros(C))
        # self.beta_a = nn.Parameter(torch.zeros(C))
        # Note that the total number of trainable parameters between
        # two convolutional capsule layer types is 4*S
        # which are stated at https://openreview.net/forum?id=HJWLfGWRb&noteId=r17t2UIgf
        #self.weights = nn.Parameter(torch.randn(1, K*K*B, C, P, P))
        self.weights = nn.Parameter(torch.randn(1,1,C,B+1))
        # op
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=2)
        self.conv1 = BasicConv2d_activation(C*K*K*16, C*K*K*1, 1)       
        self.conv2 = BasicConv2d_activationRL(C*K*K*17, C*17, 1)   
        # self.gn = nn.GroupNorm(self.B,self.B*17)
        #self.conv3 = BasicConv2d_activation()
        self.gamma = nn.Parameter(torch.ones(1, 1, 1,B,1 ))

       
    

    def add_pathes(self, x, B, K, psize, stride):
        """
            Shape:
                Input:     (b, H, W, B*(P*P+1))
                Output:    (b, H', W', K, K, B*(P*P+1))
        """
        b, h, w, c = x.shape
        # keep the scale
        # y= self.gamma
        # print("gama",y.shape)
        x_padding_h = torch.zeros(b, 1, w, c).cuda()
        x_padding_w = torch.zeros(b, h+2, 1, c).cuda()
        x = torch.cat([ x_padding_h, x, x_padding_h ], dim=1)
        x = torch.cat([ x_padding_w, x, x_padding_w ], dim=2)
        b, h, w, c = x.shape
        #assert h == w
        assert c == B*(psize+1)
        oh = ow = int(((h - K )/stride)+ 1) # moein - changed from: oh = ow = int((h - K + 1) / stride)
        idxs = [[(h_idx + k_idx) \
                for k_idx in range(0, K)] \
                for h_idx in range(0, h - K + 1, stride)]
        
        x = x[:, idxs, :, :]
        x = x[:, :, :, idxs, :]
        # print("addpath",x.shape)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        # print("addpath-1",x.shape)
        return x, oh, ow


    def transform_view(self, p,a, w, P,b,oh,ow, w_shared=False):
        """
            For conv_caps:
                Input:     (b*H*W, K*K*B, P*P)
                Output:    (b*H*W, K*K*B, C, P*P)
            For class_caps:
                Input:     (b, H*W*B, P*P)
                Output:    (b, H*W*B, C, P*P)
        """
        # print(p)
        # print(a.shape)
        gamma = self.gamma
        mu = p.mean(-1, keepdim=True)
        var = p.var(-1, keepdim=True)
        # b, B, psize = x.shape
        # print("mu",mu.shape)
        # print("var",var.shape)
        p = ((p-mu) / (var+self.eps).sqrt())
        p = p.permute(0,2,1).reshape(b,oh*ow, self.K*self.K,self.B, self.psize)
        p = a*p*self.gamma
        # print("p",p.shape)
        gamma = F.softmax( gamma,dim =3)
        psum = torch.sum(p*gamma, dim=3, keepdim=True) #[128, 256, 9, 1, 16]
        # print("psum",psum.shape)
        p = torch.cat([p,psum],dim=3)
        # print("pcat",p.shape)
        
        w = w.repeat(b,oh*ow, self.K*self.K, 1, 1)
        # print("w",w.shape)
        w = F.softmax(w,dim =3)
        # print("w",w.shape)
        # print("p",p.shape)
        # exit()??
        p= torch.matmul(w,p) #[128, 256, 9, 48, 16]
        p = p.reshape(b*oh*ow,self.K*self.K*self.C,P*P)

        return p

    def transform_view2(self, x, w, P, w_shared=False):
        """
            For conv_caps:
                Input:     (b*H*W, K*K*B, P*P)
                Output:    (b*H*W, K*K*B, C, P*P)
            For class_caps:
                Input:     (b, H*W*B, P*P)
                Output:    (b, H*W*B, C, P*P)
        """
        
        b, B, psize = x.shape
        assert psize == P*P
        x = x.view(b, B, P, P)
        if w_shared:
            hw = int(B / w.size(1))
            w = w.repeat(1, hw, 1, 1, 1)
     
        x = x.repeat(1, 1, 1, 1)
        v = x 
        v = v.view(b, B, P*P)
        return v

    def add_coord(self, v, b, h, w, B, C, psize):
        """
            Shape:
                Input:     (b, H*W*B, C, P*P)
                Output:    (b, H*W*B, C, P*P)
        """
        #assert h == w
        v = v.view(b, h, w, B, C, psize)
        coor = torch.arange(h, dtype=torch.float32) / h
        coor_h = torch.cuda.FloatTensor(1, h, 1, 1, 1, self.psize).fill_(0.)
        coor_w = torch.cuda.FloatTensor(1, 1, w, 1, 1, self.psize).fill_(0.)
        coor_h[0, :, 0, 0, 0, 0] = coor
        coor_w[0, 0, :, 0, 0, 1] = coor
        v = v + coor_h + coor_w
        v = v.view(b, h*w*B, C, psize)
        return v

    def forward(self, x):
        b, c, h, w = x.shape
        # print(b, h, w, c)
        # x = self.gn(x)
        x = x.permute(0, 2, 3, 1)
        
        if not self.w_shared:
            # add patches
            x, oh, ow = self.add_pathes(x, self.B, self.K, self.psize, self.stride)

            # transform view
            p_in = x[:, :, :, :, :, :self.B*self.psize].contiguous()
            a_in = x[:, :, :, :, :, self.B*self.psize:].contiguous()
            # print("11",p_in.shape)
            # print("21",a_in.shape)
            p_in = p_in.view(b,oh*ow, self.K*self.K,self.B, self.psize).permute(0,3,1,2,4).reshape(b,self.B,-1)
            a_in = a_in.view(b,oh*ow, self.K*self.K,self.B, 1)

            # # p_in = p_in*a_in
            # print("1",p_in.shape)
            # print("2",a_in.shape)
            # exit()
            p_out = self.transform_view(p_in,a_in, self.weights, self.P,b,oh,ow)
            #p_out = F.dropout(p_out, p = 0.5)
            
            p_out_R = p_out.reshape(p_out.size(0), -1)
            p_out_R = p_out_R.reshape(b, oh, ow, -1)
            p_out = p_out_R
            #p_out_R = p_out_R.reshape(b, oh, ow, self.B*self.K*self.K, self.P*self.P)
            p_out_1 = p_out_R.permute(0, 3, 1, 2)

            a_out = self.conv1(p_out_1)
        
            
            #a_out = self.sigmoid(torch.norm(p_out_R, dim=4))
            a_out = a_out.permute(0, 2, 3, 1)
            
            out = torch.cat([p_out, a_out], dim=3)
            out = out.permute(0, 3, 1, 2)
         
            
            out = self.conv2(out)
       
            #out = out.permute(0, 2, 3, 1)
        else:
            # print(c,self.B,self.psize)
            assert c == self.B*(self.psize+1)
            assert 1 == self.K
            assert 1 == self.stride
            #x = x.reshape(b, 1, 1, h*w*c)
            #b, h, w, c = x.shape
            p_in = x[:, :, :, :self.B*self.psize].contiguous()
            #a_in = x[:, :, :, :, :, self.B*self.psize:].contiguous()
            p_in = p_in.view(b, h*w*self.B, self.psize)
            #a_in = a_in.view(b*oh*ow, self.K*self.K*self.B, 1)
            
            p_out = self.transform_view2(p_in, self.weights, self.P)
            #p_out = F.dropout(p_out, p = 0.5)
            p_out_R = p_out.reshape(p_out.size(0), -1)
            p_out_R = p_out_R.reshape(b, h, w, -1)
            p_out = p_out_R
            p_out_R = p_out_R.reshape(b, h, w, self.B*self.K*self.K, self.P*self.P)
            p_out_R = p_out_R.reshape(b, h, w, -1)
            p_in_T = p_out_R.permute(0, 3, 1, 2)
            
            a_out = self.conv1(p_in_T)
            #out = self.sigmoid(torch.norm(p_out_R, dim=4))
            out = a_out.reshape(b, -1)

        #return out, p_out, a_out
        return out


class CapsNet(nn.Module):
    """A network with one ReLU convolutional layer followed by
    a primary convolutional capsule layer and two more convolutional capsule layers.

    Suppose image shape is 28x28x1, the feature maps change as follows:
    1. ReLU Conv1
        (_, 1, 28, 28) -> 5x5 filters, 32 out channels, stride 2 with padding
        x -> (_, 32, 14, 14)
    2. PrimaryCaps
        (_, 32, 14, 14) -> 1x1 filter, 32 out capsules, stride 1, no padding
        x -> pose: (_, 14, 14, 32x4x4), activation: (_, 14, 14, 32)
    3. ConvCaps1
        (_, 14, 14, 32x(4x4+1)) -> 3x3 filters, 32 out capsules, stride 2, no padding
        x -> pose: (_, 6, 6, 32x4x4), activation: (_, 6, 6, 32)
    4. ConvCaps2
        (_, 6, 6, 32x(4x4+1)) -> 3x3 filters, 32 out capsules, stride 1, no padding
        x -> pose: (_, 4, 4, 32x4x4), activation: (_, 4, 4, 32)
    5. ClassCaps
        (_, 4, 4, 32x(4x4+1)) -> 1x1 conv, 10 out capsules
        x -> pose: (_, 10x4x4), activation: (_, 10)

        Note that ClassCaps only outputs activation for each class

    Args:
        A: output channels of normal conv
        B: output channels of primary caps
        C: output channels of 1st conv caps
        D: output channels of 2nd conv caps
        E: output channels of class caps (i.e. number of classes)
        K: kernel of conv caps
        P: size of square pose matrix
        iters: number of EM iterations
        ...
    """
    def __init__(self, A=64, B=64, C=64, D=64, E=10, FF=32, G=32, K=3, P=4, iters=3):
        super(CapsNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=A,
                               kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm2d(num_features=A, eps=0.001,
                                 momentum=0.1, affine=True)
        self.bn2 = nn.BatchNorm2d(num_features=C*17, eps=0.001,
                                 momentum=0.1, affine=True)
        self.bn3 = nn.BatchNorm2d(num_features=C*17, eps=0.001,
                                 momentum=0.1, affine=True)
        self.bn4 = nn.BatchNorm2d(num_features=D*17, eps=0.001,
                                 momentum=0.1, affine=True)      
        self.bn5 = nn.BatchNorm2d(num_features=FF*17, eps=0.001,
                                 momentum=0.1, affine=True)   
        # self.bn6 = nn.BatchNorm2d(num_features=G*17, eps=0.001,
        #                          momentum=0.1, affine=True)        
        self.relu1 = nn.ReLU(inplace=False)
        self.primary_caps1 = PrimaryCaps(A, B, 1, P, stride=2)
        self.primary_caps2 = PrimaryCaps(C*17, C, 1, P, stride=2)
        self.primary_caps3 = PrimaryCaps(C*17, D, 1, P, stride=2)
        self.primary_caps4 = PrimaryCaps(D*17, FF, 1, P, stride=2)
        #self.primary_caps5 = PrimaryCaps(FF*17, E, 1, P, stride=1)
        self.primary_caps5 = PrimaryCapsClass(FF*17, E, 1, P, stride=1, h=2, w=2)
        self.conv_caps1_1 = ConvCaps(B, C, K, P, stride=1, iters=iters)
        self.conv_caps1_2 = ConvCaps(C, C, K, P, stride=1, iters=iters)
        self.conv_caps2_1 = ConvCaps(C, D, K, P, stride=1, iters=iters)
        self.conv_caps2_2 = ConvCaps(C, D, K, P, stride=1, iters=iters)
        self.conv_caps3_1 = ConvCaps(D, D, K, P, stride=1, iters=iters)
        self.conv_caps3_2 = ConvCaps(D, D, K, P, stride=1, iters=iters)
        self.conv_caps4_1 = ConvCaps(FF, G, K, P, stride=1, iters=iters)
        self.conv_caps4_2 = ConvCaps(FF, G, K, P, stride=1, iters=iters)
        self.class_caps = ConvCaps(E, E, 1, P, stride=1, iters=iters,
                                        coor_add=True, w_shared=True) 
        # self.convd = RF(A, A)
        self.gelu4 = nn.GELU()
        self.gelu2 = nn.GELU()
        self.gelu3 = nn.GELU()
        self.gelu5 = nn.GELU()
        self.gelu1 = nn.GELU()
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        #x = self.convd(x)
        x = self.relu1(x)
        x = self.primary_caps1(x)
        x = self.conv_caps1_1(x)
        # print("cap1",x.shape)
        # x = x.permute(0, 2, 3, 1)
        x = self.conv_caps1_2(x)
        x = self.bn2(x)
        x = self.gelu2(x)
        # print(x.shape)
        x = self.primary_caps2(x)
        x = self.conv_caps2_1(x)
        # x = x.permute(0, 2, 3, 1)
        x = self.conv_caps2_2(x)
        x = self.bn3(x)
        x = self.gelu3(x)
        x = self.primary_caps3(x)

        x = self.conv_caps3_1(x)

        # x = x.permute(0, 2, 3, 1)
        x = self.conv_caps3_2(x)

        x = self.bn4(x)
        x = self.gelu4(x)
        x = self.primary_caps4(x)
      
        
        x = self.conv_caps4_1(x)
        # x = x.permute(0, 2, 3, 1)
        x = self.conv_caps4_2(x)    
        x = self.bn5(x)
        x = self.gelu5(x)
        x = self.primary_caps5(x)
        # print(x.shape)
        
        x = self.class_caps(x)
        
        return x


def capsules(**kwargs):
    """Constructs a CapsNet model.
    """
    model = CapsNet(**kwargs)
    return model


'''
TEST
Run this code with:
```
python -m capsules.py
```
'''
if __name__ == '__main__':
    model = capsules(E=10)
