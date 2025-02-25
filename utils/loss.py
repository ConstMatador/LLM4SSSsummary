from torch import nn, squeeze
from numpy import sqrt
from torch.nn.modules.distance import PairwiseDistance

class ScaledL2Loss(nn.Module):
    def __init__(self, len_series: int, len_reduce: int):
        super(ScaledL2Loss, self).__init__()

        self.l2 = PairwiseDistance(p=2).cuda()
        self.l1 = PairwiseDistance(p=1).cuda()
        self.scale_factor_original = sqrt(len_series)
        self.scale_factor_reduce = sqrt(len_reduce)


    def forward(self, one, another, one_reduce, another_reduce):
        # one: (batch_size, len_series), another: (batch_size, len_reduce)
        
        original_l2 = self.l2(one, another) / self.scale_factor_original    # (batch_size)
        reduce_l2 = self.l2(one_reduce, another_reduce) / self.scale_factor_reduce    # (batch_size)
        
        return self.l1(original_l2.reshape(1, -1), reduce_l2.reshape(1, -1))[0] / one.shape[0]    
        # (1, batch_size) -> scalar