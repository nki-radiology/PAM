"""
ViT-PAM for big network: we do not use the Transformer directly to the CT scan (1, 1, 192, 192, 300), but to the feature-maps extracted through the encoder layers of the DefNet.
So we have "volumes" of dimensions (1, 256, 12, 12, 18) = (B, C, H, W, L).
"""

import torch
import torch.nn             as nn
import torch.nn.functional  as F
from   einops               import rearrange
import os
from   general_config       import visiontransformer, deformation

"""
We define the Patch_size = P and the number of patches that will be vectorized = N. 
N = (H*W*L)/(P*P*P)
So, having P = 6, we obtain N = 12 and each patch is actually in a space of dimension P*P*P*C, considering that we have C layers. 
This number, P*P*P*C (=13824 in our case), is called embedding size. But we do not use this number as embedding size, since it's too large, so we use instead
embedding_size = 4096.

In addition to the PatchEmbedding it's needed also the PositionEmbedding, which is useful to retain position information. 
"""

class PatchEmbedding(nn.Module):
    def __init__(self, 
                 in_channels: int = deformation.filters[4],                         # in_channels = filters[4] --> 64 or 256
                 patch_size : int= 6,        
                 emb_size   : int = visiontransformer.emb_size,
                 img_size   : int = [12,12,18]                                      # img_size = size of each layer of the feature-map
                ):
        super(PatchEmbedding, self).__init__()
        self.projection = nn.Conv3d(in_channels, emb_size, kernel_size = patch_size, stride = patch_size)
        
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_size))
        n_patches = (img_size[0]*img_size[1]*img_size[2] // patch_size**3)
        self.positions = nn.Parameter(torch.randn(1, n_patches, emb_size))
        
    def forward(self, x):
        b, _, _, _, _ = x.shape
        x = self.projection(x)
        x = x.flatten(2)
        x = x.transpose(-1, -2)                         # (B, n_patches, emb_size) 

        
        #cls_tokens = repeat(self.cls_token, '() n e -> b n e', b = b)
        #x = torch.cat([cls_tokens, x], dim = 1) 
        
        x += self.positions        
        return x 
    
class MultiHeadAttention(nn.Module):
    def __init__(self, 
                 emb_size  : int = visiontransformer.emb_size,
                 num_heads : int = visiontransformer.ViT_heads,     # default = 16 
                 dropout   : float = 0
                ):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.head_dim = emb_size // num_heads

        assert (
            self.head_dim * num_heads == emb_size
        ), "Embedding size needs to be divisible by heads"
        
        self.qkv = nn.Linear(emb_size, emb_size*3)   # queries, keys, values matrix
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)
        
    def forward(self, x):
                # Split keys, queries and vales in num_heads
        qkv = rearrange(self.qkv(x), "b n (h d qkv) -> (qkv) b h n d", h = self.num_heads, qkv = 3)
        queries, keys, values = qkv[0], qkv[1], qkv[2]
        
        # Sum up over the last axis
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)
        
        """
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))
        """
        
        scaling_factor = self.emb_size**(1/2)
        att = F.softmax(energy, dim =-1)/scaling_factor
        att = self.att_drop(att)
        
        out = torch.einsum('bhal, bhlv -> bhav', att, values)   # sum over the third axis
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        
        return out
    
class TransformerBlock(nn.Module):
    def __init__(self, 
                 emb_size           : int = visiontransformer.emb_size, 
                 num_heads          : int = visiontransformer.ViT_heads, 
                 dropout            : float = 0., 
                 forward_expansion  : int = 4
                ):
        super().__init__()
        self.attention = MultiHeadAttention(emb_size, num_heads, dropout)
        self.norm1 = nn.LayerNorm(emb_size)
        self.norm2 = nn.LayerNorm(emb_size)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(emb_size, forward_expansion*emb_size),
            nn.ReLU(),
            nn.Linear(forward_expansion*emb_size, emb_size)
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self,x):
        """ The Transformer Block operates the normalization before the Self-Attention and before the MLP. 
            This choice should help the gradients.
            """
        h = x
        # Normalization of the input
        x = self.norm1(x)
        # Self-attention to the normalized input
        x = self.attention(x)
        # Concatenation of input and self-attended 
        x = x + h
        
        h = x 
        # Normalization of the previous output
        x = self.norm2(x)
        # MLP to the normalized x
        x = self.feed_forward(x)
        # Concatenation of the MLP output and MLP input
        x = x + h
        
        return x
        
        
class Transformer(nn.Module):
    def __init__(self, 
                 emb_size     : int = visiontransformer.emb_size,
                 in_channels  : int = deformation.filters[4], 
                 patch_size   : int = 6, 
                 num_heads    : int = visiontransformer.ViT_heads,
                 num_layers   : int = visiontransformer.ViT_layers,    # number of repetition of the TransformerBlock, default = 12
                ):
        super().__init__()
        self.patch_emb = PatchEmbedding(in_channels, patch_size, emb_size)        # patch embeddings and position embeddings
        self.TransformerBlock = TransformerBlock(emb_size, num_heads)
        
        tblocks = []
        for i in range(num_layers):
            tblocks.append(self.TransformerBlock)
        self.tblocks = nn.Sequential(*tblocks)
        
    def forward(self,x):
        x = self.patch_emb(x)
        x = self.tblocks(x)
        return x
        
"""
    # To summarize the complete model
from torchsummary import summary
model  = Transformer()
print(model)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model  = model.to(device)
summary = summary(model, [(256, 12, 12, 18)], device='cuda')
""" 
    