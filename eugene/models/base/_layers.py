import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat, rearrange

# ACTIVATIONS -- Layers that apply a non-linear activation function
class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return input

class Exponential(nn.Module):
    __constants__ = ['inplace']
    inplace: bool

    def __init__(self, inplace: bool = False):
        super(Exponential, self).__init__()
        self.inplace = inplace

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return torch.exp(input)

    def extra_repr(self) -> str:
        inplace_str = 'inplace=True' if self.inplace else ''
        return inplace_str

    
ACTIVATION_REGISTRY = {
    "relu": nn.ReLU,
    "leaky_relu": nn.LeakyReLU,
    "gelu": nn.GELU,
    "elu": nn.ELU,
    "sigmoid": nn.Sigmoid,
    "tanh": nn.Tanh,
    "softplus": nn.Softplus,
    "identity": Identity,
    "exponential": Exponential
}

# CONVOLUTIONS -- Layers that convolve the input
class BiConv1D(nn.Module):
	def __init__(
		self, 
		in_channels, 
		out_channels, 
		kernel_size, 
		stride=1, 
		padding="same", 
		dilation=1, 
		groups=1, 
		bias=True,
		dropout_rate=0.0, 
		device=None,
		dtype=None,
	):
		super(BiConv1D, self).__init__()
		self.in_channels = in_channels
		self.out_channels = out_channels
		self.kernel_size = kernel_size
		self.stride = stride
		self.padding = padding
		self.dilation = dilation
		self.groups = groups
		self.weight = nn.init.xavier_uniform_(nn.Parameter(torch.zeros(out_channels, in_channels, kernel_size)))
		if bias:   
			self.bias = nn.Parameter(torch.zeros(out_channels))
		else:
			self.bias = None
		if dropout_rate != 0.0 and dropout_rate is not None:
			self.dropout_rate = dropout_rate
		else:
			self.dropout_rate = None
			
	def forward(self, x):
		x_fwd = F.conv1d(x, self.weight, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups)  		
		x_rev = F.conv1d(x, torch.flip(self.weight, dims=[0, 1]), stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups)
		if self.bias is not None:
			x_fwd = torch.add(x_fwd.transpose(1, 2), self.bias).transpose(1, 2)
			x_rev = torch.add(x_rev.transpose(1, 2), self.bias).transpose(1, 2)
		if self.dropout_rate is not None:
			x_fwd = F.dropout(F.relu(x_fwd), p=self.dropout_rate)
			x_rev = F.dropout(F.relu(x_rev), p=self.dropout_rate)
		return torch.add(x_fwd, x_rev)


	def __repr__(self):
		return "BiConv1D({}, {}, kernel_size={}, stride={}, padding={}, dilation={}, groups={}, bias={})".format(
			self.in_channels, self.out_channels, self.kernel_size, self.stride, self.padding, self.dilation, self.groups, self.bias is not None
		)


CONVOLUTION_REGISTRY = {
	"conv1d": nn.Conv1d,
	"biconv1d": BiConv1D,
}

# POOLERS -- Layers that reduce the size of the input
POOLING_REGISTRY = {
	"max": nn.MaxPool1d,
	"avg": nn.AvgPool1d,
}

# RECURRENCES -- Layers that can be used in a recurrent context
RECURRENT_REGISTRY = {
	"rnn": nn.RNN,
	"lstm": nn.LSTM,
	"gru": nn.GRU
}

class MultiHeadAttention(nn.Module):

    def __init__(
        self, 
        input_dim: int, 
        head_dim: int, 
        num_heads: int = 1,
        dropout_rate: float = 0.0, 
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.projection_dim = self.num_heads * self.head_dim
        self.need_projection = not ((self.projection_dim == self.input_dim) and (self.num_heads == 1)) 
        self.dropout_rate = dropout_rate
        self.scale_factor = head_dim ** -0.5
        self.qkv = nn.Linear(
            self.input_dim, 
            self.projection_dim * 3, 
            bias = False
        )
        self.softmax = nn.Softmax(dim = -1)
        self.dropout_layer = nn.Dropout(self.dropout_rate)
        self.projection_layer = nn.Sequential(
            nn.Linear(self.projection_dim, self.input_dim), 
            nn.Dropout(self.dropout_rate)
        ) if self.need_projection else nn.Identity()
        
    def forward(self, x, mask):
        qkv = self.qkv(x).chunk(3, dim = -1)  #qkv is a tuple of tensors - need to map to extract individual q,k,v
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.num_heads), qkv)  
        
        scaled_score = torch.matmul(q, k.transpose(-1, -2)) * self.scale_factor
        
        if mask is not None: 
            mask = mask.unsqueeze(1).expand(x.size(0), q.size(2), k.size(2)) # [b,n] --> [b,1,n] --> [b,n,n]
            mask = mask.unsqueeze(1).repeat(1, self.heads, 1, 1) #Tell Zhu-Li we did the thing: [b,n,n] --> [b,h,n,n]    
            scaled_score = scaled_score.masked_fill(mask, torch.finfo(torch.float32).min)
            
        attention = self.softmax(scaled_score)
        attention = self.dropout_layer(attention)
        
        output = torch.matmul(attention, v)
        output = rearrange(output, 'b h n d -> b n (h d)')
        output = self.projection_layer(output)
        return output

# ATTENTIONS -- Layers that apply an attention mechanism
TRANSFORMER_REGISTRY = {
	"MHA": MultiHeadAttention,
}

# NORMALIZERS -- Layers that normalize the input
NORMALIZER_REGISTRY = {
	"batchnorm": nn.BatchNorm1d,
	"layernorm": nn.LayerNorm,
}

# WRAPPERS -- Layers that wrap other layers
class Residual(nn.Module):
	def __init__(self, module):
		super(Residual, self).__init__()
		self.wrapped = module

	def forward(self, x):
		return x + self.wrapped(x)

WRAPPER_REGISTRY = {
	"residual": Residual
}

# GLUERS -- Layers that go in between other layers
# from yuzu
class Flatten(nn.Module):
	def __init__(self):
		super(Flatten, self).__init__()

	def forward(self, x):
		return x.permute(0, 2, 1).contiguous().view(x.shape[0], -1)

# from yuzu
class Unsqueeze(nn.Module):
	def __init__(self, dim):
		super(Unsqueeze, self).__init__()
		self.dim = dim

	def forward(self, x):
		return x.unsqueeze(self.dim)

class View(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def __repr__(self):
        return f'View{self.shape}'

    def forward(self, input):
        '''
        Reshapes the input according to the shape saved in the view data structure.
        '''
        batch_size = input.size(0)
        shape = (batch_size, *self.shape)
        out = input.view(shape)
        return out
		
GLUER_REGISTRY = {
	"flatten": Flatten,
	"unsqueeze": Unsqueeze,
	"view": View
}

class Clip(nn.Module):
	def __init__(self, min, max):
		super().__init__()
		self.min = min
		self.max = max

	def forward(self, x):
		return torch.clamp(x, self.min, self.max)

MISC_REGISTRY = {
	"clip": Clip
}


class MultiHeadAttentionLayer(nn.Module):

    def __init__(
        self, 
        input_dim: int, 
        head_dim: int, 
        num_heads: int = 1,
        dropout_rates: float = 0.0, 
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.projection_dim = self.num_heads * self.head_dim
        need_projection = not ((self.projection_dim == self.input_dim) and (self.num_heads == 1)) 
        self.dropout_rates = dropout_rates

        self.scale_factor = head_dim ** -0.5
        self.qkv = nn.Linear(
            self.input_dim, 
            self.projection_dim * 3, 
            bias = False
        )
        
        self.softmax = nn.Softmax(dim = -1)
        self.dropout_layer = nn.Dropout(self.dropout_rates)
        
        self.projection_layer = nn.Sequential(
            nn.Linear(self.projection_dim, self.input_dim), 
            nn.Dropout(self.dropout_rates)
        ) if need_projection else nn.Identity()
        
    def forward(self, x, mask):
        qkv = self.qkv(x).chunk(3, dim = -1)  #qkv is a tuple of tensors - need to map to extract individual q,k,v
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.num_heads), qkv)  
        
        scaled_score = torch.matmul(q, k.transpose(-1, -2)) * self.scale_factor
        
        if mask is not None: 
            mask = mask.unsqueeze(1).expand(x.size(0), q.size(2), k.size(2)) # [b,n] --> [b,1,n] --> [b,n,n]
            mask = mask.unsqueeze(1).repeat(1, self.heads, 1, 1) #Tell Zhu-Li we did the thing: [b,n,n] --> [b,h,n,n]    
            scaled_score = scaled_score.masked_fill(mask, torch.finfo(torch.float32).min)
            
        attention = self.softmax(scaled_score)
        attention = self.dropout_layer(attention)
        
        output = torch.matmul(attention, v)
        output = rearrange(output, 'b h n d -> b n (h d)')
        output = self.projection_layer(output)
        return output