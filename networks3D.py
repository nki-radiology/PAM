import torch
import torch.functional    as     F
from   torch               import nn 
from   torch.autograd      import Variable
from   layer3D             import Conv3dBlock
from   layer3D             import ResBlocks
from   layer3D             import MLP
from   spatial_transformer import SpatialTransformer
try:
    from itertools import izip as zip
except ImportError:
    pass

# ============================================== Style Encoder Class ==============================================
#
#                            The Style encoder is also known as appearance encoder
#
# =================================================================================================================

class StyleEncoder(nn.Module):
    def __init__(self, n_downsample, input_dim, dim, style_dim, norm, activ, pad_type):
        super(StyleEncoder, self).__init__()
        self.model  = []
        self.model += [Conv3dBlock(input_dim, dim, 7, 1, 3, norm=norm, activation=activ, pad_type=pad_type)]
        # input_dim, output_dim, kernel_size, stride, padding=0, norm='none', activation='relu', pad_type='zero
        
        for i in range(2):
            self.model += [Conv3dBlock(dim, 2 * dim, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type)]
            dim *= 2
        for i in range(n_downsample - 2):
            self.model += [Conv3dBlock(dim, dim, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type)]
            
        self.model += [nn.AdaptiveAvgPool3d(1)] # global average pooling
        self.model += [nn.Conv3d(dim, style_dim, 1, 1, 0)]
        self.model = nn.Sequential(*self.model)
        self.output_dim = dim

    def forward(self, x):
        return self.model(x)
    
    
    
# ============================================== Content Encoder Class ==============================================
#
#                                The Content encoder is also known as shape encoder
#
# ===================================================================================================================

class ContentEncoder(nn.Module):
    def __init__(self, n_downsample, n_res, input_dim, dim, norm, activ, pad_type):
        super(ContentEncoder, self).__init__()
        self.model = []
        self.model += [Conv3dBlock(input_dim, dim, 7, 1, 3, norm=norm, activation=activ, pad_type=pad_type)]
        # downsampling blocks
        for i in range(n_downsample):
            self.model += [Conv3dBlock(dim, 2 * dim, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type)]
            dim *= 2
        # residual blocks
        self.model += [ResBlocks(n_res, dim, norm=norm, activation=activ, pad_type=pad_type)]
        self.model = nn.Sequential(*self.model)
        self.output_dim = dim

    def forward(self, x):
        return self.model(x)


# ============================================== Decoder Class ==============================================
#
#                                This decoder class belongs to the generator module
#
# =========================================================================================================== 

class Decoder(nn.Module):
    def __init__(self, n_upsample, n_res, dim, output_dim, res_norm='adain', activ='relu', pad_type='zero'):
        super(Decoder, self).__init__()

        self.model = []
        # AdaIN residual blocks
        self.model += [ResBlocks(n_res, dim, res_norm, activ, pad_type=pad_type)]
        # upsampling blocks
        for i in range(n_upsample):
            self.model += [nn.Upsample(scale_factor=2),
                           Conv3dBlock(dim, dim // 2, 5, 1, 2, norm='ln', activation=activ, pad_type=pad_type)]
            dim //= 2
        # use reflection padding in the last conv layer
        self.model += [Conv3dBlock(dim, output_dim, 7, 1, 3, norm='none', activation='tanh', pad_type=pad_type)]
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x)


# ============================================= Generator Class =============================================
#
#                                        Generator based on the AdaIn model
#
# =========================================================================================================== 

class AdaINGen(nn.Module):
    # AdaIN auto-encoder architecture
    def __init__(self, input_dim, params):
        super(AdaINGen, self).__init__()
        dim          = params['dim']
        style_dim    = params['style_dim']
        n_downsample = params['n_downsample']
        n_res        = params['n_res']
        activ        = params['activ']
        pad_type     = params['pad_type']
        mlp_dim      = params['mlp_dim']

        # style encoder
        self.enc_style   = StyleEncoder(4, input_dim, dim, style_dim, norm='none', activ=activ, pad_type=pad_type)

        # content encoder
        self.enc_content = ContentEncoder(n_downsample, n_res, input_dim, dim, 'in', activ, pad_type=pad_type)
        self.dec         = Decoder(n_downsample, n_res, self.enc_content.output_dim, input_dim, res_norm='adain', activ=activ, pad_type=pad_type)

        # MLP to generate AdaIN parameters
        self.mlp         = MLP(style_dim, self.get_num_adain_params(self.dec), mlp_dim, 3, norm='none', activ=activ)
        
        # Spataial Transformer Network
        self.spatial_transformer = SpatialTransformer(tuple(self.input_dim))
        

    def forward(self, fixed, moving):
        # reconstruct an image
        content_fx, style_fake_fx = self.encode(fixed)
        content_mv, style_fake_mv = self.encode(moving)
        fixed_recon               = self.decode(content_fx, style_fake_fx)
        moving_recon              = self.decode(content_mv, style_fake_mv)
        
        # Spatial Transformer
        transformation            = self.decode(content_fx, content_mv)
        elastic_registered_image = self.spatial_transformer(moving, transformation)
        return elastic_registered_image

    def encode(self, images):
        # encode an image to its content and style codes
        style_fake = self.enc_style(images)
        content    = self.enc_content(images)
        return content, style_fake

    def decode(self, content, style):
        # decode content and style codes to an image
        adain_params = self.mlp(style)
        self.assign_adain_params(adain_params, self.dec)
        images = self.dec(content)
        return images

    def assign_adain_params(self, adain_params, model):
        # assign the adain_params to the AdaIN layers in model
        for m in model.modules():
            if m.__class__.__name__ == "AdaptiveInstanceNorm3d":
                mean = adain_params[:, :m.num_features]
                std  = adain_params[:, m.num_features:2*m.num_features]
                m.bias   = mean.contiguous().view(-1)
                m.weight = std.contiguous().view(-1)
                if adain_params.size(1) > 2*m.num_features:
                    adain_params = adain_params[:, 2*m.num_features:]

    def get_num_adain_params(self, model):
        # return the number of AdaIN parameters needed by the model
        num_adain_params = 0
        for m in model.modules():
            if m.__class__.__name__ == "AdaptiveInstanceNorm3d":
                num_adain_params += 2*m.num_features
        return num_adain_params
    


# =========================================== Discriminator Class ===========================================
#
#                                       Discriminator based on the AdaIn model
#
# ===========================================================================================================

class MsImageDis(nn.Module):
    # Multi-scale discriminator architecture
    def __init__(self, input_dim, params):
        super(MsImageDis, self).__init__()
        self.n_layer    = params['n_layer']
        self.gan_type   = params['gan_type']
        self.dim        = params['dim']
        self.norm       = params['norm']
        self.activ      = params['activ']
        self.num_scales = params['num_scales']
        self.pad_type   = params['pad_type']
        self.input_dim  = input_dim
        self.downsample = nn.AvgPool3d(3, stride=2, padding=[1, 1], count_include_pad=False)
        self.cnns       = nn.ModuleList()
        for _ in range(self.num_scales):
            self.cnns.append(self._make_net())

    def _make_net(self):
        dim    = self.dim
        cnn_x  = []
        cnn_x += [Conv3dBlock(self.input_dim, dim, 4, 2, 1, norm='none', activation=self.activ, pad_type=self.pad_type)]
        for i in range(self.n_layer - 1):
            cnn_x += [Conv3dBlock(dim, dim * 2, 4, 2, 1, norm=self.norm, activation=self.activ, pad_type=self.pad_type)]
            dim *= 2
        cnn_x += [nn.Conv3d(dim, 1, 1, 1, 0)]
        cnn_x  = nn.Sequential(*cnn_x)
        return cnn_x

    def forward(self, x):
        outputs = []
        for model in self.cnns:
            outputs.append(model(x))
            x = self.downsample(x)
        return outputs

    def calc_dis_loss(self, input_fake, input_real):
        # calculate the loss to train D
        outs0 = self.forward(input_fake)
        outs1 = self.forward(input_real)
        loss  = 0

        for it, (out0, out1) in enumerate(zip(outs0, outs1)):
            if self.gan_type == 'lsgan':
                loss += torch.mean((out0 - 0)**2) + torch.mean((out1 - 1)**2)
            elif self.gan_type == 'nsgan':
                all0  = Variable(torch.zeros_like(out0.data).cuda(), requires_grad=False)
                all1  = Variable(torch.ones_like(out1.data).cuda(), requires_grad=False)
                loss += torch.mean(F.binary_cross_entropy(F.sigmoid(out0), all0) +
                                   F.binary_cross_entropy(F.sigmoid(out1), all1))
            else:
                assert 0, "Unsupported GAN type: {}".format(self.gan_type)
        return loss

    def calc_gen_loss(self, input_fake):
        # calculate the loss to train G
        outs0 = self.forward(input_fake)
        loss = 0
        for it, (out0) in enumerate(outs0):
            if self.gan_type == 'lsgan':
                loss += torch.mean((out0 - 1)**2) # LSGAN
            elif self.gan_type == 'nsgan':
                all1  = Variable(torch.ones_like(out0.data).cuda(), requires_grad=False)
                loss += torch.mean(F.binary_cross_entropy(F.sigmoid(out0), all1))
            else:
                assert 0, "Unsupported GAN type: {}".format(self.gan_type)
        return 