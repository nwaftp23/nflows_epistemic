import numpy as np
import torch
from torch import nn

from nflows.transforms.base import Transform

class Planar(Transform):
    """
    Planar flow as introduced in arXiv: 1505.05770
        f(z) = z + u * h(w * z + b)
    """

    def __init__(self, dim, act="leaky_relu", context_dims=0, context_net=nn.Identity):
        """
        Constructor of the planar flow
        :param shape: shape of the latent variable z
        :param h: nonlinear function h of the planar flow (see definition of f above)
        :param u,w,b: optional initialization for parameters
        """
        super().__init__()
        lim_w = np.sqrt(2. / np.prod(dim))
        lim_u = np.sqrt(2)

        if context_dims > 0:
            self.transform_net = context_net(context_dims, dim*3)
        else:
            self.u = nn.Parameter(torch.empty([dim])[None])
            nn.init.uniform_(self.u, -lim_u, lim_u)
            self.w = nn.Parameter(torch.empty([dim])[None])
            nn.init.uniform_(self.w, -lim_w, lim_w)
            self.b = nn.Parameter(torch.zeros(1))

        self.act = act
        if act == "tanh":
            self.h = torch.tanh
        elif act == "leaky_relu":
            self.h = torch.nn.LeakyReLU(negative_slope=0.2)
        else:
            raise NotImplementedError('Nonlinearity is not implemented.')


    def forward(self, z, context=None, kwargs={}):
        if context != None:
            params = self.transform_net(z, context=context, **kwargs)
            W = params[:,:1]
            U = params[:,1:2]
            B = params[:,2:]
        else:
            W = self.w
            U = self.u
            B = self.b
        lin = W*z+B
        inner = W*U
        u = U + (torch.log(1 + torch.exp(inner)) - 1 - inner) * W / torch.sum(W ** 2)
        if self.act == "tanh":
            h_ = lambda x: 1 / torch.cosh(x) ** 2
        elif self.act == "leaky_relu":
            h_ = lambda x: (x < 0) * (self.h.negative_slope - 1.0) + 1.0
        z_ = z + u * self.h(lin)
        log_det = torch.log(torch.abs(1 + W*u*h_(lin)))
        return z_, log_det.squeeze()

    def inverse(self, z, context=None, kwargs={}):
        if self.act != "leaky_relu":
            raise NotImplementedError('This flow has no algebraic inverse.')
        if context != None:
            params = self.transform_net(z, context=context, **kwargs)
            W = params[:,:1]
            U = params[:,1:2]
            B = params[:,2:]
        else:
            W = self.w
            U = self.u
            B = self.b
        lin = W*z+B
        inner = W*U
        u = U + (torch.log(1 + torch.exp(inner)) - 1 - inner) * W / torch.sum(W ** 2)
        a = ((lin+B) / (1 + inner) < 0) * (self.h.negative_slope - 1.0) + 1.0  
        u = a*(U + (torch.log(1 + torch.exp(inner)) - 1 - inner) * W / torch.sum(W ** 2))
        z_ = z - 1 / (1 + inner) * (lin + u * B)
        log_det = -torch.log(torch.abs(1 + W*u))
        #if log_det.dim() == 0:
        #    log_det = log_det.unsqueeze(0)
        if log_det.dim() == 1:
            log_det = log_det.unsqueeze(1)
        return z_, log_det.squeeze()
