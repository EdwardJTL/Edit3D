import math
import torch.nn as nn
import torch


class PosEmbedding(nn.Module):
    def __init__(self,
                 max_logscale,
                 N_freqs,
                 logscale=True,
                 multi_pi=False,):
        """
        Defines a function that embeds x to (x, sin(2^k x), cos(2^k x), ...)
        """
        super(PosEmbedding, self).__init__()

        self.N_freqs = N_freqs
        self.funcs = [torch.sin, torch.cos]

        if logscale:
            self.freqs = 2 ** torch.linspace(0, max_logscale, N_freqs)
        else:
            self.freqs = torch.linspace(0, 2 ** max_logscale, N_freqs)

        if multi_pi:
            self.freqs = self.freqs * math.pi

        return

    def get_out_dim(self):
        return 3 + 3 * 2 * self.N_freqs

    def forward(self, x):
        """
        Inputs:
            x: (B, 3)

        Outputs:
            out: (B, 6*N_freqs+3)
        """
        out = [x]
        for freq in self.freqs:
            for func in self.funcs:
                out += [func(freq * x)]

        return torch.cat(out, -1)
