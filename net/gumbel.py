import torch
import torch.nn as nn

class Gumbel(nn.Module):
    ''' 
    Returns differentiable discrete outputs. Applies a Gumbel-Softmax trick on every element of x. 
    '''
    def __init__(self, config):
        super(Gumbel, self).__init__()
        self.factor = config.GUMBEL_FACTOR
        self.gumbel_noise = config.GUMBEL_NOISE

    def forward(self, x):
        if not self.training:  # no Gumbel noise during inference
            return (x >= 0).float()

        if self.gumbel_noise:
            U = torch.rand_like(x)
            g= -torch.log( - torch.log(U + 1e-8) + 1e-8)
            x = x + g

        soft = torch.sigmoid(x / self.factor)
        hard = ((soft >= 0.5).float() - soft).detach() + soft
        assert not torch.any(torch.isnan(hard))
        
        return hard