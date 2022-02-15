import torch

def BCHW_to_BHWC(x):
    return x.permute(0, 2, 3, 1)

def BWHC_to_BCHW(x):
    return x.permute(0, 3, 1, 2)

#adapted from the open-mmlab implementation of swin transformer
class Windower(torch.nn.Module):
    def __init__(self, window_size=(7, 7), shift_size=(0, 0)):
        super().__init__()
        self.window_size = window_size
        self.shift_size = shift_size
        self.neg_shift_size = (-shift_size[0], -shift_size[1])
        self.do_shift = (shift_size[0] + shift_size[1]) != 0

    def forward(self, x):
        B, H, W, C = x.shape
        if self.do_shift:
            x = torch.roll(x, shifts=self.neg_shift_size, dims=(1, 2))
        Wh, Ww = self.window_size
        x = x.view(B, H // Wh, Wh, W // Ww, Ww, C)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        windows = windows.view(B, -1, Wh, Ww, C)
        return windows

    def undo(self, windows, H, W):
        B, nW, Wh, Ww, C = windows.shape
        x = windows.view(B, H // Wh, W // Ww, Wh, Ww, C)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        x = x.view(B, H, W, C)
        if self.do_shift:
            x = torch.roll(x, shifts=self.shift_size, dims=(1, 2))
        return x
