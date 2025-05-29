import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    #"MeanPooling",
    "AttentiveStatisticsPooling"
]


class Pooling(nn.Module):
    def __init__(self):
        super().__init__()
    def compute_length_from_mask(self, mask):
        """
        mask: (batch_size, T)
        Assuming that the sampling rate is 16kHz, the frame shift is 20ms
        """
        wav_lens = torch.sum(mask, dim=1) # (batch_size, )
        feat_lens = torch.div(wav_lens-1, 16000*0.02, rounding_mode="floor") + 1
        feat_lens = feat_lens.int().tolist()
        return feat_lens
        
    def forward(self, x, mask):
        raise NotImplementedError
    

class AttentiveStatisticsPooling(Pooling):
    """
    AttentiveStatisticsPooling
    Paper: Attentive Statistics Pooling for Deep Speaker Embedding
    Link: https://arxiv.org/pdf/1803.10963.pdf
    """
    def __init__(self, input_size):
        super().__init__()
        if isinstance(input_size, tuple):
            input_size = input_size[0]
        self._indim = input_size
        self.sap_linear = nn.Linear(input_size, input_size)
        self.attention = nn.Parameter(torch.FloatTensor(input_size, 1))
        torch.nn.init.normal_(self.attention, mean=0, std=1)
        
    def forward(self, xs, mask):
        """
        xs: (batch_size, T, feat_dim)
        mask: (batch_size, T)
        """
        feat_lens = self.compute_length_from_mask(mask)
        pooled_list = []
        for x, feat_len in zip(xs, feat_lens):
            x = x[:feat_len].unsqueeze(0)
            #print(f"Before sap_linear, x.shape: {x.shape}")  # Debug
            h = torch.tanh(self.sap_linear(x))
            #print(f"After sap_linear, h.shape: {h.shape}")  # Debug
            w = torch.matmul(h, self.attention).squeeze(dim=2)
            w = F.softmax(w, dim=1).view(x.size(0), x.size(1), 1)
            mu = torch.sum(x * w, dim=1)
            rh = torch.sqrt((torch.sum((x**2) * w, dim=1) - mu**2).clamp(min=1e-5))
            x = torch.cat((mu, rh), 1).squeeze(0)
            pooled_list.append(x)
        return torch.stack(pooled_list)


