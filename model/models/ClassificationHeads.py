import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class SwinHead(nn.Module):
    """Classification head.

    Args:
        num_classes (int): Number of classes to be classified.
        in_channels (int): Number of channels in input feature.
        loss_cls (dict): Config for building loss.
            Default: dict(type='CrossEntropyLoss')
        spatial_type (str): Pooling type in spatial dimension. Default: 'avg'.
        dropout_ratio (float): Probability of dropout layer. Default: 0.5.
        init_std (float): Std value for Initiation. Default: 0.01.
        kwargs (dict, optional): Any keyword argument to be used to initialize
            the head.
    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 spatial_type='avg',
                 dropout_ratio=0.5,
                 init_std=0.01,
                 **kwargs):
        super().__init__()

        self.num_classes=num_classes
        self.in_channels=in_channels
        self.spatial_type = spatial_type
        self.dropout_ratio = dropout_ratio
        self.init_std = init_std
        if self.dropout_ratio != 0:
            self.dropout = nn.Dropout(p=self.dropout_ratio)
        else:
            self.dropout = None
        self.fc_cls = nn.Linear(self.in_channels, self.num_classes)

        if self.spatial_type == 'avg':
            # use `nn.AdaptiveAvgPool3d` to adaptively match the in_channels.
            self.avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        else:
            self.avg_pool = None

    def init_weights(self):
        """Initiate the parameters from scratch."""
        nn.init.normal(self.fc_cls, std=self.init_std)

    def forward(self, x):
        """Defines the computation performed at every call.

        Args:
            x (torch.Tensor): The input data.

        Returns:
            torch.Tensor: The classification scores for input samples.
        """
        # [N, in_channels, 4, 7, 7]
        if self.avg_pool is not None:
            x = self.avg_pool(x)
        # [N, in_channels, 1, 1, 1]
        if self.dropout is not None:
            x = self.dropout(x)
        # [N, in_channels, 1, 1, 1]
        x = x.view(x.shape[0], -1)
        # [N, in_channels]
        cls_score = self.fc_cls(x)
        # [N, num_classes]
        return cls_score
    
class C3DHead(nn.Module):
    """Classification head.

    Args:
        num_classes (int): Number of classes to be classified.
        in_channels (int): Number of channels in input feature.
        loss_cls (dict): Config for building loss.
            Default: dict(type='CrossEntropyLoss')
        spatial_type (str): Pooling type in spatial dimension. Default: 'avg'.
        dropout_ratio (float): Probability of dropout layer. Default: 0.5.
        init_std (float): Std value for Initiation. Default: 0.01.
        kwargs (dict, optional): Any keyword argument to be used to initialize
            the head.
    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 dropout_ratio=0.5,
                 init_std=0.01,
                 **kwargs):
        super().__init__()

        self.num_classes=num_classes
        self.in_channels=in_channels
        self.dropout_ratio = dropout_ratio
        self.init_std = init_std
        if self.dropout_ratio != 0:
            self.dropout = nn.Dropout(p=self.dropout_ratio)
        else:
            self.dropout = None

        self.fc1 = nn.Linear(self.in_channels, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, self.num_classes)

    def init_weights(self):
        """Initiate the parameters from scratch."""
        nn.init.normal(self.fc1, std=self.init_std)
        nn.init.normal(self.fc2, std=self.init_std)
        nn.init.normal(self.fc3, std=self.init_std)

    def forward(self, x):
        """Defines the computation performed at every call.

        Args:
            x (torch.Tensor): The input data.

        Returns:
            torch.Tensor: The classification scores for input samples.
        """
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        cls_score = self.fc3(x)

        return cls_score
    
class R3DHead(nn.Module):
    """Classification head.

    Args:
        num_classes (int): Number of classes to be classified.
        in_channels (int): Number of channels in input feature.
        loss_cls (dict): Config for building loss.
            Default: dict(type='CrossEntropyLoss')
        spatial_type (str): Pooling type in spatial dimension. Default: 'avg'.
        dropout_ratio (float): Probability of dropout layer. Default: 0.5.
        init_std (float): Std value for Initiation. Default: 0.01.
        kwargs (dict, optional): Any keyword argument to be used to initialize
            the head.
    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 dropout_ratio=0.5,
                 init_std=0.01,
                 **kwargs):
        super().__init__()

        self.num_classes=num_classes
        self.in_channels=in_channels
        self.dropout_ratio = dropout_ratio
        self.init_std = init_std
        if self.dropout_ratio != 0:
            self.dropout = nn.Dropout(p=self.dropout_ratio)
        else:
            self.dropout = None

        self.fc1 = nn.Linear(self.in_channels, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, self.num_classes)

    def init_weights(self):
        """Initiate the parameters from scratch."""
        nn.init.normal(self.fc1, std=self.init_std)
        nn.init.normal(self.fc2, std=self.init_std)
        nn.init.normal(self.fc3, std=self.init_std)

    def forward(self, x):
        """Defines the computation performed at every call.

        Args:
            x (torch.Tensor): The input data.

        Returns:
            torch.Tensor: The classification scores for input samples.
        """
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        cls_score = self.fc3(x)

        return cls_score

class TimeSFormerHead(nn.Module):
    """Classification head.

    Args:
        num_classes (int): Number of classes to be classified.
        in_channels (int): Number of channels in input feature.
        loss_cls (dict): Config for building loss.
            Default: dict(type='CrossEntropyLoss')
        spatial_type (str): Pooling type in spatial dimension. Default: 'avg'.
        dropout_ratio (float): Probability of dropout layer. Default: 0.5.
        init_std (float): Std value for Initiation. Default: 0.01.
        kwargs (dict, optional): Any keyword argument to be used to initialize
            the head.
    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 dropout_ratio=0.5,
                 init_std=0.01,
                 **kwargs):
        super().__init__()

        self.num_classes=num_classes
        self.in_channels=in_channels
        self.dropout_ratio = dropout_ratio
        self.init_std = init_std
        if self.dropout_ratio != 0:
            self.dropout = nn.Dropout(p=self.dropout_ratio)
        else:
            self.dropout = None

        self.fc1 = nn.Linear(self.in_channels, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, self.num_classes)

    def init_weights(self):
        """Initiate the parameters from scratch."""
        nn.init.normal(self.fc1, std=self.init_std)
        nn.init.normal(self.fc2, std=self.init_std)
        nn.init.normal(self.fc3, std=self.init_std)

    def forward(self, x):
        """Defines the computation performed at every call.

        Args:
            x (torch.Tensor): The input data.

        Returns:
            torch.Tensor: The classification scores for input samples.
        """
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        cls_score = self.fc3(x)

        return cls_score
    
def pretrain_cls_transforms(x:torch.Tensor,pretrain_scheme:str,n_cells=None):
    if pretrain_scheme.upper()=="CUBICPUZZLE3D":
        return rearrange(x,'(B N) H -> B (N H)',N=n_cells)
    elif pretrain_scheme.upper()=="SEPARATEJIGSAWPUZZLE3D":
        return x
    elif pretrain_scheme.upper()=="JOINTJIGSAWPUZZLE3D":
        return x

class SwinPretrainHead(nn.Module):
    """Classification head.

    Args:
        num_classes (int): Number of classes to be classified.
        in_channels (int): Number of channels in input feature.
        loss_cls (dict): Config for building loss.
            Default: dict(type='CrossEntropyLoss')
        spatial_type (str): Pooling type in spatial dimension. Default: 'avg'.
        dropout_ratio (float): Probability of dropout layer. Default: 0.5.
        init_std (float): Std value for Initiation. Default: 0.01.
        kwargs (dict, optional): Any keyword argument to be used to initialize
            the head.
    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 spatial_type='avg',
                 dropout_ratio=0.5,
                 init_std=0.01,
                 n_cells=None,
                 pretrain_scheme=None,
                 **kwargs):
        super().__init__()

        self.num_classes=num_classes
        self.in_channels=in_channels
        self.spatial_type = spatial_type
        self.dropout_ratio = dropout_ratio
        self.pretrain_scheme=pretrain_scheme
        self.n_cells=n_cells
        self.init_std = init_std
        if self.dropout_ratio != 0:
            self.dropout = nn.Dropout(p=self.dropout_ratio)
        else:
            self.dropout = None
        self.fc_cls = nn.Linear(self.in_channels, self.num_classes)

        if self.spatial_type == 'avg':
            # use `nn.AdaptiveAvgPool3d` to adaptively match the in_channels.
            self.avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        else:
            self.avg_pool = None

    def init_weights(self):
        """Initiate the parameters from scratch."""
        nn.init.normal(self.fc_cls, std=self.init_std)

    def forward(self, x):
        """Defines the computation performed at every call.

        Args:
            x (torch.Tensor): The input data.

        Returns:
            torch.Tensor: The classification scores for input samples.
        """
        # [N, in_channels, 4, 7, 7]
        if self.avg_pool is not None:
            x = self.avg_pool(x)
        # [N, in_channels, 1, 1, 1]
        if self.dropout is not None:
            x = self.dropout(x)
        # [N, in_channels, 1, 1, 1]
        x = x.view(x.shape[0], -1)

        x=pretrain_cls_transforms(x,pretrain_scheme=self.pretrain_scheme,n_cells=self.n_cells)

        # [N, in_channels]
        cls_score = self.fc_cls(x)
        # [N, num_classes]
        return cls_score
    
class C3DPretrainHead(nn.Module):
    """Classification head.

    Args:
        num_classes (int): Number of classes to be classified.
        in_channels (int): Number of channels in input feature.
        loss_cls (dict): Config for building loss.
            Default: dict(type='CrossEntropyLoss')
        spatial_type (str): Pooling type in spatial dimension. Default: 'avg'.
        dropout_ratio (float): Probability of dropout layer. Default: 0.5.
        init_std (float): Std value for Initiation. Default: 0.01.
        kwargs (dict, optional): Any keyword argument to be used to initialize
            the head.
    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 dropout_ratio=0.5,
                 init_std=0.01,
                 n_cells=None,
                 pretrain_scheme=None,
                 **kwargs):
        super().__init__()

        self.num_classes=num_classes
        self.in_channels=in_channels
        self.dropout_ratio = dropout_ratio
        self.pretrain_scheme=pretrain_scheme
        self.n_cells=n_cells
        self.init_std = init_std
        if self.dropout_ratio != 0:
            self.dropout = nn.Dropout(p=self.dropout_ratio)
        else:
            self.dropout = None

        self.fc = nn.Linear(self.in_channels, self.num_classes)

    def init_weights(self):
        """Initiate the parameters from scratch."""
        nn.init.normal(self.fc, std=self.init_std)

    def forward(self, x):
        """Defines the computation performed at every call.

        Args:
            x (torch.Tensor): The input data.

        Returns:
            torch.Tensor: The classification scores for input samples.
        """
        x=pretrain_cls_transforms(x,pretrain_scheme=self.pretrain_scheme,n_cells=self.n_cells)
        cls_score = self.fc(x)

        return cls_score

class R3DPretrainHead(nn.Module):
    """Classification head.

    Args:
        num_classes (int): Number of classes to be classified.
        in_channels (int): Number of channels in input feature.
        loss_cls (dict): Config for building loss.
            Default: dict(type='CrossEntropyLoss')
        spatial_type (str): Pooling type in spatial dimension. Default: 'avg'.
        dropout_ratio (float): Probability of dropout layer. Default: 0.5.
        init_std (float): Std value for Initiation. Default: 0.01.
        kwargs (dict, optional): Any keyword argument to be used to initialize
            the head.
    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 dropout_ratio=0.5,
                 init_std=0.01,
                 n_cells=None,
                 pretrain_scheme=None,
                 **kwargs):
        super().__init__()

        self.num_classes=num_classes
        self.in_channels=in_channels
        self.dropout_ratio = dropout_ratio
        self.pretrain_scheme=pretrain_scheme
        self.n_cells=n_cells
        self.init_std = init_std
        if self.dropout_ratio != 0:
            self.dropout = nn.Dropout(p=self.dropout_ratio)
        else:
            self.dropout = None

        self.fc = nn.Linear(self.in_channels, self.num_classes)

    def init_weights(self):
        """Initiate the parameters from scratch."""
        nn.init.normal(self.fc, std=self.init_std)

    def forward(self, x):
        """Defines the computation performed at every call.

        Args:
            x (torch.Tensor): The input data.

        Returns:
            torch.Tensor: The classification scores for input samples.
        """
        x=pretrain_cls_transforms(x,pretrain_scheme=self.pretrain_scheme,n_cells=self.n_cells)
        cls_score = self.fc(x)

        return cls_score
    
class TimeSFormerPretrainHead(nn.Module):
    """Classification head.

    Args:
        num_classes (int): Number of classes to be classified.
        in_channels (int): Number of channels in input feature.
        loss_cls (dict): Config for building loss.
            Default: dict(type='CrossEntropyLoss')
        spatial_type (str): Pooling type in spatial dimension. Default: 'avg'.
        dropout_ratio (float): Probability of dropout layer. Default: 0.5.
        init_std (float): Std value for Initiation. Default: 0.01.
        kwargs (dict, optional): Any keyword argument to be used to initialize
            the head.
    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 dropout_ratio=0.5,
                 init_std=0.01,
                 n_cells=None,
                 pretrain_scheme=None,
                 **kwargs):
        super().__init__()

        self.num_classes=num_classes
        self.in_channels=in_channels
        self.dropout_ratio = dropout_ratio
        self.pretrain_scheme=pretrain_scheme
        self.n_cells=n_cells
        self.init_std = init_std
        if self.dropout_ratio != 0:
            self.dropout = nn.Dropout(p=self.dropout_ratio)
        else:
            self.dropout = None

        self.fc = nn.Linear(self.in_channels, self.num_classes)

    def init_weights(self):
        """Initiate the parameters from scratch."""
        nn.init.normal(self.fc, std=self.init_std)

    def forward(self, x):
        """Defines the computation performed at every call.

        Args:
            x (torch.Tensor): The input data.

        Returns:
            torch.Tensor: The classification scores for input samples.
        """
        x=pretrain_cls_transforms(x,pretrain_scheme=self.pretrain_scheme,n_cells=self.n_cells)
        cls_score = self.fc(x)

        return cls_score