import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import BaseModel, BasicFullyConnectedModule, BasicConv1D
from .base._utils import GetFlattenDim


mode_dict = {"dna": 1, "rbp": 2}


class DeepBind(BaseModel):
    """
    DeepBind model implemented from Alipanahi et al 2015 in PyTorch

    DeepBind is a model that takes in a DNA or RNA sequence and outputs a probability of 
    binding for a given DNA transcription factor or RNA binding protein respectively.
    This is a flexible implementation of the original DeepBind architecture that allows users
    to modify the number of convolutional layers, the number of fully connected layers, and
    many more hyperparameters. If parameters for the CNN and FCN are not passed in, the model
    will be instantiated with the parameters described in Alipanahi et al 2015.

    Like the original DeepBind models, this model can be used for both DNA and RNA binding. For DNA,
    we implemented the "dna" mode which only uses the max pooling of the representation generated by 
    the convolutional layers. For RNA, we implemented the "rbp" mode which uses both the max and average
    pooling of the representation generated by the convolutional layers.
    - For "ss" models, we use the representation generated by the convolutional layers and pass that through a 
        set of fully connected layer to generate the output.
    - For "ds" models, we use the representation generated by the convolutional layers for both the forward and 
        reverse complement strands and pass that through the same set of fully connected layers to generate the output.
    - For "ts" models, we use the representation generated by separate sets of convolutional layers for the forward and
        reverse complement strands and passed that through separate sets of fully connected layers to generate the output.
    
    aggr defines how the output for "ds" and "ts" models is generated. If "max", we take the max of the forward and reverse
    complement outputs. If "avg", we take the average of the forward and reverse complement outputs. There is no "concat" for
    the current implementation of DeepBind models.

    Parameters
    ----------
    input_len : int
        Length of input sequence
    output_dim : int
        Number of output classes
    mode : str
        Mode of model, either "dna" or "rbp"
    strand : str
        Strand of model, either "ss", "ds", or "ts"
    task : str
        Task of model, either "regression" or "classification"
    aggr : str
        Aggregation method of model, either "max" or "avg"
    loss_fxn : str
        Loss function of model, either "mse" or "cross_entropy"
    optimizer : str
        Optimizer of model, either "adam" or "sgd"
    lr : float
        Learning rate of model
    scheduler : str
        Scheduler of model, either "lr_scheduler" or "plateau"
    scheduler_patience : int
        Scheduler patience of model
    mp_kwargs : dict
        Keyword arguments for multiprocessing
    conv_kwargs : dict
        Keyword arguments for convolutional layers
    fc_kwargs : dict
        Keyword arguments for fully connected layers
    """
    def __init__(
        self,
        input_len: int,
        output_dim: int,
        strand: str = "ss",
        task: str = "regression",
        aggr: str = "max",
        loss_fxn: str ="mse",
        mode: str = "rbp",
        conv_kwargs: dict = {},
        fc_kwargs: dict = {},
        **kwargs
    ):
        super().__init__(
            input_len, 
            output_dim, 
            strand=strand, 
            task=task, 
            aggr=aggr, 
            loss_fxn=loss_fxn, 
            **kwargs
        )
        self.conv_kwargs, self.fc_kwargs = self.kwarg_handler(conv_kwargs, fc_kwargs)
        self.mode = mode
        self.mode_multiplier = mode_dict[self.mode]
        self.aggr = aggr
        self.convnet = BasicConv1D(input_len=input_len, **self.conv_kwargs)
        self.pool_dim = GetFlattenDim(self.convnet.module, seq_len=input_len)
        self.max_pool = nn.MaxPool1d(kernel_size=self.pool_dim)
        self.avg_pool = nn.AvgPool1d(kernel_size=self.pool_dim)
        if self.strand == "ss":
            self.fcn = BasicFullyConnectedModule(
                input_dim=self.convnet.out_channels * self.mode_multiplier,
                output_dim=output_dim,
                **self.fc_kwargs
            )
        elif self.strand == "ds":
            self.fcn = BasicFullyConnectedModule(
                input_dim=self.convnet.out_channels * self.mode_multiplier,
                output_dim=output_dim,
                **self.fc_kwargs
            )
        elif self.strand == "ts":
            self.fcn = BasicFullyConnectedModule(
                self.convnet.out_channels * self.mode_multiplier,
                output_dim=output_dim,
                **self.fc_kwargs
            )
            self.reverse_convnet = BasicConv1D(
                input_len=input_len, 
                **self.conv_kwargs
                )
            self.reverse_fcn = BasicFullyConnectedModule(
                self.convnet.out_channels * self.mode_multiplier,
                output_dim=output_dim,
                **self.fc_kwargs
            )

    def forward(self, x, x_rev_comp=None):
        x = self.convnet(x)
        if self.mode == "rbp":
            x = torch.cat((self.max_pool(x), self.avg_pool(x)), dim=1)
            x = x.view(x.size(0), self.convnet.out_channels * 2)
        elif self.mode == "dna":
            x = self.max_pool(x)
            x = x.view(x.size(0), self.convnet.out_channels)
        x = self.fcn(x)
        if self.strand == "ds":
            x_rev_comp = self.convnet(x_rev_comp)
            if self.mode == "rbp":
                x_rev_comp = torch.cat((self.max_pool(x_rev_comp), self.avg_pool(x_rev_comp)), dim=1)
                x_rev_comp = x_rev_comp.view(x_rev_comp.size(0), self.convnet.out_channels * 2)
            elif self.mode == "dna":
                x_rev_comp = self.max_pool(x_rev_comp)
                x_rev_comp = x_rev_comp.view(x_rev_comp.size(0), self.convnet.out_channels)
            x_rev_comp = self.fcn(x_rev_comp)
            if self.aggr == "max":
                x = F.max_pool1d(torch.cat((x, x_rev_comp), dim=1), 2)
            elif self.aggr == "avg":
                x = torch.mean(torch.cat((x, x_rev_comp), dim=1), dim=1).unsqueeze(dim=1)
        elif self.strand == "ts":
            x_rev_comp = self.reverse_convnet(x_rev_comp)
            if self.mode == "rbp":
                x_rev_comp = torch.cat((self.max_pool(x_rev_comp), self.avg_pool(x_rev_comp)), dim=1)
                x_rev_comp = x_rev_comp.view(x_rev_comp.size(0), self.convnet.out_channels * 2)
            elif self.mode == "dna":
                x_rev_comp = self.max_pool(x_rev_comp)
                x_rev_comp = x_rev_comp.view(x_rev_comp.size(0), self.convnet.out_channels)
            x_rev_comp = self.reverse_fcn(x_rev_comp)
            if self.aggr == "max":
                x = F.max_pool1d(torch.cat((x, x_rev_comp), dim=1), 2)
            elif self.aggr == "avg":
                x = torch.mean(torch.cat((x, x_rev_comp), dim=1), dim=1).unsqueeze(dim=1)
        return x

    def kwarg_handler(self, conv_kwargs, fc_kwargs):
        """Sets default kwargs for conv and fc modules if not specified"""
        conv_kwargs.setdefault("channels", [4, 16])
        conv_kwargs.setdefault("conv_kernels", [16])
        conv_kwargs.setdefault("omit_final_pool", True)
        conv_kwargs.setdefault("dropout_rates", 0.25)
        conv_kwargs.setdefault("batchnorm", False)
        fc_kwargs.setdefault("hidden_dims", [32])
        fc_kwargs.setdefault("dropout_rate", 0.25)
        fc_kwargs.setdefault("batchnorm", False)
        return conv_kwargs, fc_kwargs


class DeepSEA(BaseModel):
    """DeepSEA model implementation for EUGENe
    
    Default parameters are those specified in the DeepSEA paper. We currently do not implement a "ds" or "ts" model
    for DeepSEA.

    Parameters
    ----------
    input_len:
        int, input sequence length
    channels:
        list-like or int, channel width for each conv layer. If int each of the three layers will be the same channel width
    conv_kernels:
        list-like or int, conv kernel size for each conv layer. If int will be the same for all conv layers
    pool_kernels:
        list-like or int, maxpooling kernel size for the first two conv layers. If int will be the same for all conv layers
    dropout_rates:
        list-like or float, dropout rates for each conv layer. If int will be the same for all conv layers
    """
    def __init__(
        self,
        input_len: int = 1000,
        output_dim: int = 1,
        strand: str = "ss",
        task: str = "regression",
        aggr: str = None,
        loss_fxn: str = "mse",
        conv_kwargs: dict = {},
        fc_kwargs: dict = {},
        **kwargs
    ):
        super().__init__(
            input_len, 
            output_dim, 
            strand=strand, 
            task=task, 
            aggr=aggr, 
            loss_fxn=loss_fxn, 
            **kwargs
        )
        self.conv_kwargs, self.fc_kwargs = self.kwarg_handler(conv_kwargs, fc_kwargs)
        self.convnet = BasicConv1D(
            input_len=input_len, 
            **self.conv_kwargs)
        self.fcn = BasicFullyConnectedModule(
            input_dim=self.convnet.flatten_dim, 
            output_dim=output_dim, 
            **self.fc_kwargs
        )

    def forward(self, x, x_rev_comp=None):
        x = self.convnet(x)
        x = x.view(x.size(0), self.convnet.flatten_dim)
        x = self.fcn(x)
        return x

    def kwarg_handler(self, conv_kwargs, fc_kwargs):
        """Sets default kwargs for conv and fc modules if not specified"""
        conv_kwargs.setdefault("channels", [4, 320, 480, 960])
        conv_kwargs.setdefault("conv_kernels", [8, 8, 8])
        conv_kwargs.setdefault("pool_kernels", [4, 4, 4])
        conv_kwargs.setdefault("omit_final_pool", True)
        conv_kwargs.setdefault("activation", "relu")
        conv_kwargs.setdefault("dropout_rates", [0.2, 0.2, 0.5])
        conv_kwargs.setdefault("batchnorm", False)
        fc_kwargs.setdefault("hidden_dims", [925])
        return conv_kwargs, fc_kwargs
