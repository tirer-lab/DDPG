import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import math
from scipy.interpolate import interp2d
import torch.fft
from functools import partial


class A_functions_fft:
    """
    All input vectors are of shape (Batch, ...).
    All output vectors are of shape (Batch, DataDimension) ---- to remain compatible with code
    """

    def singulars(self):
        """
        Returns a vector containing the singular values. The shape of the vector should be the same as the smaller dimension (like U)
        """
        raise NotImplementedError()

    def add_zeros(self, vec):
        """
        Adds trailing zeros to turn a vector from the small dimension (U) to the big dimension (V)
        """
        raise NotImplementedError()
    
    def A(self, vec):
        """
        Multiplies the input vector by A
        """
        raise NotImplementedError()
    
    def At(self, vec):
        """
        Multiplies the input vector by A transposed
        """
        raise NotImplementedError()
    
    def A_pinv_add_eta(self, vec, eta_reg = 1e-4):
        """
        Multiplies the input vector by the pseudo inverse of A
        """
        raise NotImplementedError()
    
    def invAAt(self, vec):
        """
        Multiplies the input vector by the inverse of AAt
        """
        raise NotImplementedError()    
    




#Deblurring
class Deblurring_fft(A_functions_fft):

    def __init__(self, kernel, channels, img_dim, device, eta_reg = 1e-4):
        self.img_dim = img_dim
        self.channels = channels
        if kernel.dim() == 1:
            kernel2D = torch.matmul(kernel[:,None],kernel[None,:])/torch.sum(kernel)**2
            self.kernel = kernel2D
        else:
            self.kernel = kernel
        self.eta_reg = eta_reg

    def A(self,vec):
        I = vec.reshape(vec.shape[0], self.channels, self.img_dim, self.img_dim)
        out = torch.zeros(I.shape[0], I.shape[1], I.shape[2], I.shape[3]).to(vec.device)
        for ch in range(out.shape[1]):
            out[:,ch:ch+1, :, :] = cconv2_by_fft2(I[:, ch:ch+1, :, :], self.kernel, vec.device, flag_invertB=0)
        return out.reshape(vec.shape[0], -1)
    
    def At(self,vec):
        I = vec.reshape(vec.shape[0], self.channels, self.img_dim, self.img_dim)
        out = torch.zeros(I.shape[0], I.shape[1], I.shape[2], I.shape[3]).to(vec.device)
        flipped_kernel = torch.fliplr(torch.flipud(torch.conj(self.kernel)))
        for ch in range(out.shape[1]):
            out[:,ch:ch+1, :, :] = cconv2_by_fft2(I[:, ch:ch+1, :, :], flipped_kernel, vec.device, flag_invertB=0)
        return out.reshape(vec.shape[0], -1)
    
    def A_pinv_add_eta(self, vec, eta_reg =None):
        I = vec.reshape(vec.shape[0], self.channels, self.img_dim, self.img_dim)
        out = torch.zeros(I.shape[0], I.shape[1], I.shape[2], I.shape[3]).to(vec.device)
        if eta_reg is None:
            eta_reg = self.eta_reg

        flipped_kernel = torch.fliplr(torch.flipud(torch.conj(self.kernel)))
        for ch in range(out.shape[1]):
            temp = cconv2_invAAt_by_fft2(I[:, ch:ch+1, :, :], self.kernel, vec.device, eta=eta_reg)
            out[:,ch:ch+1, :, :] = cconv2_by_fft2(temp, flipped_kernel, vec.device, flag_invertB=0)
        return out.reshape(vec.shape[0], -1)    
    
    def invAAt(self, vec, eta_reg = None):
        I = vec.reshape(vec.shape[0], self.channels, self.img_dim, self.img_dim)
        out = torch.zeros(I.shape[0], I.shape[1], I.shape[2], I.shape[3]).to(vec.device)
        if eta_reg is None:
            eta_reg = self.eta_reg

        for ch in range(out.shape[1]):
            out[:,ch:ch+1, :, :] = cconv2_invAAt_by_fft2(I[:, ch:ch+1, :, :], self.kernel, vec.device, eta=eta_reg)
        return out.reshape(vec.shape[0], -1)

    def AtA_add_eta_inv(self, vec, eta_reg=None):  ## same as AAt_add_eta_inv
        I = vec.reshape(vec.shape[0], self.channels, self.img_dim, self.img_dim)
        out = torch.zeros(I.shape[0], I.shape[1], I.shape[2], I.shape[3]).to(vec.device)
        if eta_reg is None:
            eta_reg = self.eta_reg

        for ch in range(out.shape[1]):
            out[:,ch:ch+1, :, :] = cconv2_invAAt_by_fft2(I[:, ch:ch+1, :, :], self.kernel, vec.device, eta=eta_reg)
        return out.reshape(vec.shape[0], -1)




#Super-resolution via FFT
class Superres_fft(A_functions_fft):

    def __init__(self, kernel, channels, img_dim, device, stride = 1, eta_reg = 1e-4):
        self.img_dim = img_dim
        self.channels = channels
        if kernel.dim() == 1:
            kernel2D = torch.matmul(kernel[:,None],kernel[None,:])/torch.sum(kernel)**2
            self.kernel = kernel2D
        else:
            self.kernel = kernel

        self.ratio = stride
        small_dim = img_dim // stride
        self.small_dim = small_dim
        self.eta_reg = eta_reg

    def A(self,vec):
        I = vec.reshape(vec.shape[0], self.channels, self.img_dim, self.img_dim)
        temp = torch.zeros(vec.shape[0], self.channels, self.img_dim, self.img_dim).to(vec.device)
        for ch in range(self.channels):
            temp[:,ch:ch+1, :, :] = cconv2_by_fft2(I[:, ch:ch+1, :, :], self.kernel, vec.device, flag_invertB=0)
        out = downsample(temp, self.ratio)
        return out.reshape(vec.shape[0], -1)
    
    def At(self,vec):
        I = vec.reshape(vec.shape[0], self.channels, self.small_dim, self.small_dim)
        out = torch.zeros(I.shape[0], self.channels, self.img_dim, self.img_dim).to(vec.device)
        flipped_kernel = torch.fliplr(torch.flipud(torch.conj(self.kernel)))
        temp = upsample_MN(I,self.ratio, self.img_dim, self.img_dim)
        for ch in range(self.channels):
            out[:,ch:ch+1, :, :] = cconv2_by_fft2(temp[:, ch:ch+1, :, :], flipped_kernel, vec.device, flag_invertB=0)
        return out.reshape(vec.shape[0], -1)

    def invAAt(self, vec, eta_reg = None):
        I = vec.reshape(vec.shape[0], self.channels, self.small_dim, self.small_dim)
        out = torch.zeros(I.shape[0], I.shape[1], I.shape[2], I.shape[3]).to(vec.device)
        if eta_reg is None:
            eta_reg = self.eta_reg

        mk, nk = self.kernel.shape[:2]
        bigK = torch.zeros((self.img_dim, self.img_dim)).to(vec.device)
        bigK[:mk,:nk] = self.kernel
        bigK = torch.roll(bigK, (-int((mk-1)/2), -int((mk-1)/2)), dims=(0,1))  # pad PSF with zeros to whole image domain, and center it
        fft2_K = torch.fft.fft2(bigK)

        h0_full = torch.real(torch.fft.ifft2(torch.abs(fft2_K)**2))  # ifft( fft(filped_k) * fft(k) )
        h0 = h0_full[::self.ratio, ::self.ratio]

        fft2_h0 = torch.fft.fft2(h0)
        inv_fft2_h0 = 1 / (fft2_h0 + eta_reg)

        for ch in range(self.channels):
            out[:,ch:ch+1, :, :] = torch.real(torch.fft.ifft2(torch.fft.fft2( I[:, ch:ch+1, :, :] ) * inv_fft2_h0))

        return out.reshape(vec.shape[0], -1)

    def A_pinv_add_eta(self, vec, eta_reg = None):
        I = vec.reshape(vec.shape[0], self.channels, self.small_dim, self.small_dim)
        out = torch.zeros(I.shape[0], self.channels, self.img_dim, self.img_dim).to(vec.device)
        if eta_reg is None:
            eta_reg = self.eta_reg

        mk, nk = self.kernel.shape[:2]
        bigK = torch.zeros((self.img_dim, self.img_dim)).to(vec.device)
        bigK[:mk,:nk] = self.kernel
        bigK = torch.roll(bigK, (-int((mk-1)/2), -int((mk-1)/2)), dims=(0,1))  # pad PSF with zeros to whole image domain, and center it
        fft2_K = torch.fft.fft2(bigK)

        h0_full = torch.real(torch.fft.ifft2(torch.abs(fft2_K)**2))  # ifft( fft(filped_k) * fft(k) )
        h0 = h0_full[::self.ratio, ::self.ratio]
        fft2_h0 = torch.fft.fft2(h0)

        # fft2_h0_full = torch.abs(fft2_K)**2
        # fft2_h0_full_ = torch.stack(torch.chunk(fft2_h0_full, self.ratio, dim=0), dim=2)
        # fft2_h0_full_ = torch.cat(torch.chunk(fft2_h0_full_, self.ratio, dim=1), dim=2)
        # fft2_h0 = torch.mean(fft2_h0_full_, dim=-1, keepdim=False)

        # I_ext = upsample_MN(I ,self.ratio, self.img_dim, self.img_dim)
        # fft2_I_allChannels = torch.fft.fftn(I_ext, dim=(-2,-1))
        # fft2_I_allChannels = torch.stack(torch.chunk(fft2_I_allChannels, self.ratio, dim=2), dim=4)
        # fft2_I_allChannels = torch.cat(torch.chunk(fft2_I_allChannels, self.ratio, dim=3), dim=4)
        # fft2_I_allChannels = torch.mean(fft2_I_allChannels, dim=-1, keepdim=False)

        inv_fft2_h0 = 1 / (fft2_h0 + eta_reg)

        temp = torch.zeros(I.shape[0], I.shape[1], I.shape[2], I.shape[3]).to(vec.device)
        for ch in range(self.channels):
            temp[:,ch:ch+1, :, :] = torch.real(torch.fft.ifft2(torch.fft.fft2( I[:, ch:ch+1, :, :] ) * inv_fft2_h0))
            #temp[:,ch:ch+1, :, :] = torch.real(torch.fft.ifft2( fft2_I_allChannels[:, ch:ch+1, :, :]  * inv_fft2_h0))

        flipped_kernel = torch.fliplr(torch.flipud(torch.conj(self.kernel)))
        temp2 = upsample_MN(temp ,self.ratio, self.img_dim, self.img_dim)
        for ch in range(self.channels):
            out[:,ch:ch+1, :, :] = cconv2_by_fft2(temp2[:, ch:ch+1, :, :], flipped_kernel, vec.device, flag_invertB=0)

        return out.reshape(vec.shape[0], -1)

    def AtA_add_eta_inv(self, vec, eta_reg=None): 
        assert 1, "TODO"


def cconv2_by_fft2(A,B,device,flag_invertB=0,eta=0.01):
    # assumes that A (2D image) is bigger than B (2D kernel)

    m, n = A.shape[2:]
    mb, nb = B.shape[:2]

    # pad, multiply and transform back
    bigB = torch.zeros((m, n)).to(device)
    bigB[:mb,:nb] = B
    bigB = torch.roll(bigB, (-int((mb-1)/2), -int((mb-1)/2)), dims=(0,1))  # pad PSF with zeros to whole image domain, and center it
    #bigB = torch.from_numpy(bigB).float().to(device)

    fft2B = torch.fft.fft2(bigB)

    if flag_invertB:
        fft2B = torch.conj(fft2B) / ((torch.abs(fft2B)**2) + eta)  # Standard Tikhonov Regularization

    result = torch.real(torch.fft.ifft2(torch.fft.fft2(A) * fft2B))

    return result


def cconv2_invAAt_by_fft2(A,B,device,eta=0.01):
    # assumes that A (2D image) is bigger than B (2D kernel)

    m, n = A.shape[2:]
    mb, nb = B.shape[:2]

    # pad, multiply and transform back
    bigB = torch.zeros((m, n)).to(device)
    bigB[:mb,:nb] = B
    bigB = torch.roll(bigB, (-int((mb-1)/2), -int((mb-1)/2)), dims=(0,1))  # pad PSF with zeros to whole image domain, and center it
    #bigB = torch.from_numpy(bigB).float().to(device)

    fft2B = torch.fft.fft2(bigB)

    fft2B_norm2 = torch.abs(fft2B)**2
    inv_fft2B_norm = 1 / (fft2B_norm2 + eta)

    result = torch.real(torch.fft.ifft2(torch.fft.fft2(A) * inv_fft2B_norm))

    return result





def upsample(x, sf=3):
    '''s-fold upsampler
    Upsampling the spatial size by filling the new entries with zeros
    x: tensor image, NxCxWxH
    '''
    st = 0
    z = torch.zeros((x.shape[0], x.shape[1], x.shape[2]*sf, x.shape[3]*sf)).type_as(x)
    z[..., st::sf, st::sf].copy_(x)
    return z

def downsample(x, sf=3):
    '''s-fold downsampler
    Keeping the upper-left pixel for each distinct sfxsf patch and discarding the others
    x: tensor image, NxCxWxH
    '''
    st = 0
    return x[..., st::sf, st::sf]

def upsample_MN(x, sf=3, M=None, N=None):
    z = upsample(x, sf)
    if M is not None and N is not None:
        # make sure to keep original size after down-up
        z = z[:,:,:M,:N]
    return z


def cubic(x):
    # See Keys, "Cubic Convolution Interpolation for Digital Image
    # Processing, " IEEE Transactions on Acoustics, Speech, and Signal Processing, Vol.ASSP - 29, No. 6, December 1981, p.1155.
    absx = np.abs(x)
    absx2 = absx ** 2
    absx3 = absx ** 3
    f = (1.5 * absx3 - 2.5 * absx2 + 1) * (absx <= 1) + (-0.5 * absx3 + 2.5 * absx2 - 4 * absx + 2) * ((1 < absx) & (absx <= 2))
    return f

def prepare_cubic_filter(scale):
    # uses the kernel part of matlab's imresize function (before subsampling)
    # note: scale<1 for downsampling

    kernel_width = 4
    kernel_width = kernel_width / scale

    u = 0

    # What is the left-most pixel that can be involved in the computation?
    left = np.floor(u - kernel_width/2)

    # What is the maximum number of pixels that can be involved in the
    # computation?  Note: it's OK to use an extra pixel here; if the
    # corresponding weights are all zero, it will be eliminated at the end
    # of this function.
    P = np.ceil(kernel_width) + 1

    # The indices of the input pixels involved in computing the k-th output
    # pixel are in row k of the indices matrix.
    indices = left + np.arange(0,P,1) # = left + [0:1:P-1]

    # The weights used to compute the k-th output pixel are in row k of the
    # weights matrix.
    weights = scale * cubic(scale *(u-indices))
    weights = np.reshape(weights, [1,weights.size])
    return np.matmul(weights.T,weights)


def matlab_style_gauss2D(shape=(7,7),sigma=1.6):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m,n = [(ss-1.)/2. for ss in shape]
    y,x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h


def shift_pixel(x, sf, upper_left=True):
    """shift pixel for super-resolution with different scale factors
    Args:
        x: WxHxC or WxH
        sf: scale factor
        upper_left: shift direction
    """
    h, w = x.shape[:2]
    shift = (sf-1)*0.5
    xv, yv = np.arange(0, w, 1.0), np.arange(0, h, 1.0)
    if upper_left:
        x1 = xv + shift
        y1 = yv + shift
    else:
        x1 = xv - shift
        y1 = yv - shift

    x1 = np.clip(x1, 0, w-1)
    y1 = np.clip(y1, 0, h-1)

    if x.ndim == 2:
        x = interp2d(xv, yv, x)(x1, y1)
    if x.ndim == 3:
        for i in range(x.shape[-1]):
            x[:, :, i] = interp2d(xv, yv, x[:, :, i])(x1, y1)

    return x


