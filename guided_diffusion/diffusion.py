import os
import logging
import time
import glob

import numpy as np
import tqdm
import torch
import torch.utils.data as data

from datasets import get_dataset, data_transform, inverse_data_transform
from functions.ckpt_util import get_ckpt_path, download
from functions.ddpg_scheme import ddpg_diffusion

import torchvision.utils as tvu

from guided_diffusion.models import Model
from guided_diffusion.script_util import create_model, create_classifier, classifier_defaults, args_to_dict
import random

import lpips


loss_fn_alex = lpips.LPIPS(net='alex') # net='alex' best forward scores


def get_gaussian_noisy_img(img, noise_level):
    return img + torch.randn_like(img).cuda() * noise_level

def MeanUpsample(x, scale):
    n, c, h, w = x.shape
    out = torch.zeros(n, c, h, scale, w, scale).to(x.device) + x.view(n,c,h,1,w,1)
    out = out.view(n, c, scale*h, scale*w)
    return out

def color2gray(x):
    coef=1/3
    x = x[:,0,:,:] * coef + x[:,1,:,:]*coef +  x[:,2,:,:]*coef
    return x.repeat(1,3,1,1)

def gray2color(x):
    x = x[:,0,:,:]
    coef=1/3
    base = coef**2 + coef**2 + coef**2
    return torch.stack((x*coef/base, x*coef/base, x*coef/base), 1)    



def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    if beta_schedule == "quad":
        betas = (
            np.linspace(
                beta_start ** 0.5,
                beta_end ** 0.5,
                num_diffusion_timesteps,
                dtype=np.float64,
            )
            ** 2
        )
    elif beta_schedule == "linear":
        betas = np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":  
        betas = 1.0 / np.linspace(
            num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas


class Diffusion(object):
    def __init__(self, args, config, device=None):
        self.args = args
        self.config = config
        if device is None:
            device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        self.device = device

        self.model_var_type = config.model.var_type
        betas = get_beta_schedule(
            beta_schedule=config.diffusion.beta_schedule,
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
        )
        betas = self.betas = torch.from_numpy(betas).float().to(self.device)
        self.num_timesteps = betas.shape[0]

        alphas = 1.0 - betas
        alphas_cumprod = alphas.cumprod(dim=0)
        alphas_cumprod_prev = torch.cat(
            [torch.ones(1).to(device), alphas_cumprod[:-1]], dim=0
        )
        self.alphas_cumprod_prev = alphas_cumprod_prev
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        if self.model_var_type == "fixedlarge":
            self.logvar = betas.log()
        elif self.model_var_type == "fixedsmall":
            self.logvar = posterior_variance.clamp(min=1e-20).log()

    def sample(self, logger):
        cls_fn = None
        if self.config.model.type == 'simple':
            model = Model(self.config)

            if self.config.data.dataset == "CIFAR10":
                name = "cifar10"
            elif self.config.data.dataset == "LSUN":
                name = f"lsun_{self.config.data.category}"
            elif self.config.data.dataset == 'CelebA_HQ':
                name = 'celeba_hq'
            else:
                raise ValueError
            if name != 'celeba_hq':
                ckpt = get_ckpt_path(f"ema_{name}", prefix=self.args.exp)
                print("Loading checkpoint {}".format(ckpt))
            elif name == 'celeba_hq':
                ckpt = os.path.join(self.args.exp, "logs/celeba/celeba_hq.ckpt")
                if not os.path.exists(ckpt):
                    raise ValueError("CelebA-HQ model checkpoint not found, please download it as mentioned in README.md"
                                     " file and configure the correct path")
                    # download('https://image-editing-test-12345.s3-us-west-2.amazonaws.com/checkpoints/celeba_hq.ckpt',ckpt)
            else:
                raise ValueError
            model.load_state_dict(torch.load(ckpt, map_location=self.device))
            model.to(self.device)
            model = torch.nn.DataParallel(model)

        elif self.config.model.type == 'openai':
            config_dict = vars(self.config.model)
            model = create_model(**config_dict)
            if self.config.model.use_fp16:
                model.convert_to_fp16()
            if self.config.model.class_cond:
                ckpt = os.path.join(self.args.exp, 'logs/imagenet/%dx%d_diffusion.pt' % (
                self.config.data.image_size, self.config.data.image_size))
                if not os.path.exists(ckpt):
                    # download(
                    #     'https://openaipublic.blob.core.windows.net/diffusion/jul-2021/%dx%d_diffusion_uncond.pt' % (
                    #     self.config.data.image_size, self.config.data.image_size), ckpt)
                    raise ValueError(
                        "ImageNet model checkpoint not found, please download it as mentioned in README.md"
                        " file and configure the correct path")
            else:
                ckpt = os.path.join(self.args.exp, "logs/imagenet/256x256_diffusion_uncond.pt")
                if not os.path.exists(ckpt):
                    raise ValueError(
                        "ImageNet model checkpoint not found, please download it as mentioned in README.md"
                        " file and configure the correct path")
                    # download(
                    #     'https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion_uncond.pt',
                    #     ckpt)

            model.load_state_dict(torch.load(ckpt, map_location=self.device))
            model.to(self.device)
            model.eval()
            model = torch.nn.DataParallel(model)

            if self.config.model.class_cond:
                ckpt = os.path.join(self.args.exp, 'logs/imagenet/%dx%d_classifier.pt' % (
                self.config.data.image_size, self.config.data.image_size))
                if not os.path.exists(ckpt):
                    image_size = self.config.data.image_size
                    download(
                        'https://openaipublic.blob.core.windows.net/diffusion/jul-2021/%dx%d_classifier.pt' % image_size,
                        ckpt)
                classifier = create_classifier(**args_to_dict(self.config.classifier, classifier_defaults().keys()))
                classifier.load_state_dict(torch.load(ckpt, map_location=self.device))
                classifier.to(self.device)
                if self.config.classifier.classifier_use_fp16:
                    classifier.convert_to_fp16()
                classifier.eval()
                classifier = torch.nn.DataParallel(classifier)

                import torch.nn.functional as F
                def cond_fn(x, t, y):
                    with torch.enable_grad():
                        x_in = x.detach().requires_grad_(True)
                        logits = classifier(x_in, t)
                        log_probs = F.log_softmax(logits, dim=-1)
                        selected = log_probs[range(len(logits)), y.view(-1)]
                        return torch.autograd.grad(selected.sum(), x_in)[0] * self.config.classifier.classifier_scale

                cls_fn = cond_fn

        if self.args.inject_noise==1:
            print('Run DDPG.',
                f'Operators implementation via {self.args.operator_imp}.',
                f'{self.config.sampling.T_sampling} sampling steps.',
                f'Task: {self.args.deg}.',
                f'Noise level: {self.args.sigma_y}.'
                )
        else:
            print('Run IDPG.',
                f'Operators implementation via {self.args.operator_imp}.',
                f'{self.config.sampling.T_sampling} sampling steps.',
                f'Task: {self.args.deg}.',
                f'Noise level: {self.args.sigma_y}.'
                )
        self.ddpg_wrapper(model, cls_fn, logger)
        
        

    def ddpg_wrapper(self, model, cls_fn, logger):
        args, config = self.args, self.config

        dataset, test_dataset = get_dataset(args, config)

        device_count = torch.cuda.device_count()

        if args.subset_start >= 0 and args.subset_end > 0:
            assert args.subset_end > args.subset_start
            test_dataset = torch.utils.data.Subset(test_dataset, range(args.subset_start, args.subset_end))
        else:
            args.subset_start = 0
            args.subset_end = len(test_dataset)

        print(f'Dataset has size {len(test_dataset)}')

        def seed_worker(worker_id):
            worker_seed = args.seed % 2 ** 32
            np.random.seed(worker_seed)
            random.seed(worker_seed)

        g = torch.Generator()
        g.manual_seed(args.seed)
        val_loader = data.DataLoader(
            test_dataset,
            batch_size=config.sampling.batch_size,
            shuffle=True,
            num_workers=config.data.num_workers,
            worker_init_fn=seed_worker,
            generator=g,
        )

        # get degradation matrix
        deg = args.deg
        A_funcs = None
        if deg == 'cs_walshhadamard':
            compress_by = round(1/args.deg_scale)
            from functions.svd_operators import WalshHadamardCS
            A_funcs = WalshHadamardCS(config.data.channels, self.config.data.image_size, compress_by,
                                      torch.randperm(self.config.data.image_size ** 2, device=self.device), self.device)
        elif deg == 'cs_blockbased':
            cs_ratio = args.deg_scale
            from functions.svd_operators import CS
            A_funcs = CS(config.data.channels, self.config.data.image_size, cs_ratio, self.device)
        elif deg == 'inpainting':
            from functions.svd_operators import Inpainting
            loaded = np.load("exp/inp_masks/mask.npy")
            mask = torch.from_numpy(loaded).to(self.device).reshape(-1)
            missing_r = torch.nonzero(mask == 0).long().reshape(-1) * 3
            missing_g = missing_r + 1
            missing_b = missing_g + 1
            missing = torch.cat([missing_r, missing_g, missing_b], dim=0)
            A_funcs = Inpainting(config.data.channels, config.data.image_size, missing, self.device)
        elif deg == 'denoising':
            from functions.svd_operators import Denoising
            A_funcs = Denoising(config.data.channels, self.config.data.image_size, self.device)
        elif deg == 'colorization':
            from functions.svd_operators import Colorization
            A_funcs = Colorization(config.data.image_size, self.device)
        elif deg == 'sr_averagepooling':
            blur_by = int(args.deg_scale)
            if args.operator_imp == 'SVD':
                from functions.svd_operators import SuperResolution
                A_funcs = SuperResolution(config.data.channels, config.data.image_size, blur_by, self.device)
            else:
                raise NotImplementedError()

        elif deg == 'sr_bicubic':
            factor = int(args.deg_scale)
            def bicubic_kernel(x, a=-0.5):
                if abs(x) <= 1:
                    return (a + 2) * abs(x) ** 3 - (a + 3) * abs(x) ** 2 + 1
                elif 1 < abs(x) and abs(x) < 2:
                    return a * abs(x) ** 3 - 5 * a * abs(x) ** 2 + 8 * a * abs(x) - 4 * a
                else:
                    return 0
            k = np.zeros((factor * 4))
            for i in range(factor * 4):
                x = (1 / factor) * (i - np.floor(factor * 4 / 2) + 0.5)
                k[i] = bicubic_kernel(x)
            k = k / np.sum(k)
            kernel = torch.from_numpy(k).float().to(self.device)
            
            if args.operator_imp == 'SVD':
                from functions.svd_operators import SRConv
                A_funcs = SRConv(kernel / kernel.sum(), config.data.channels, self.config.data.image_size, self.device, stride=factor)
            elif args.operator_imp == 'FFT':                
                from functions.fft_operators import Superres_fft, prepare_cubic_filter
                k = prepare_cubic_filter(1/factor)
                kernel = torch.from_numpy(k).float().to(self.device)
                A_funcs = Superres_fft(kernel / kernel.sum(), config.data.channels, self.config.data.image_size, self.device, stride=factor)
            else:
                raise NotImplementedError()

        elif deg == 'deblur_uni':
            if args.operator_imp == 'SVD':
                from functions.svd_operators import Deblurring
                A_funcs = Deblurring(torch.Tensor([1 / 9] * 9).to(self.device), config.data.channels,
                                    self.config.data.image_size, self.device)
            elif args.operator_imp == 'FFT':
                from functions.fft_operators import Deblurring_fft
                A_funcs = Deblurring_fft(torch.Tensor([1 / 9] * 9).to(self.device), config.data.channels, self.config.data.image_size, self.device)
            else:
                raise NotImplementedError()

        elif deg == 'deblur_gauss':
            sigma = 10 # better make argument for kernel type
            pdf = lambda x: torch.exp(torch.Tensor([-0.5 * (x / sigma) ** 2]))
            kernel = torch.Tensor([pdf(-2), pdf(-1), pdf(0), pdf(1), pdf(2)]).to(self.device) # clip it as in DDRM/DDNM code, but it makes more sense to use lower sigma with the line below
            #kernel = torch.Tensor([pdf(ii) for ii in range(-30,31,1)]).to(self.device)
            if args.operator_imp == 'SVD':
                from functions.svd_operators import Deblurring
                A_funcs = Deblurring(kernel / kernel.sum(), config.data.channels, self.config.data.image_size, self.device)
            elif args.operator_imp == 'FFT':
                from functions.fft_operators import Deblurring_fft
                A_funcs = Deblurring_fft(kernel / kernel.sum(), config.data.channels, self.config.data.image_size, self.device)
            else:
                raise NotImplementedError()

        elif deg == 'deblur_aniso':
            sigma = 20
            pdf = lambda x: torch.exp(torch.Tensor([-0.5 * (x / sigma) ** 2]))
            kernel2 = torch.Tensor([pdf(-4), pdf(-3), pdf(-2), pdf(-1), pdf(0), pdf(1), pdf(2), pdf(3), pdf(4)]).to(
                self.device)
            sigma = 1
            pdf = lambda x: torch.exp(torch.Tensor([-0.5 * (x / sigma) ** 2]))
            kernel1 = torch.Tensor([pdf(-4), pdf(-3), pdf(-2), pdf(-1), pdf(0), pdf(1), pdf(2), pdf(3), pdf(4)]).to(
                self.device)

            if args.operator_imp == 'SVD':
                from functions.svd_operators import Deblurring2D
                A_funcs = Deblurring2D(kernel1 / kernel1.sum(), kernel2 / kernel2.sum(), config.data.channels,
                                    self.config.data.image_size, self.device)
            elif args.operator_imp == 'FFT':
                # unlike when using 'SVD' mode, here you can implement any 2D kernel that you want (not just seperable kernels)
                from functions.fft_operators import Deblurring_fft
                kernel = torch.matmul(kernel1[:,None],kernel2[None,:])
                A_funcs = Deblurring_fft(kernel / kernel.sum(), config.data.channels, self.config.data.image_size, self.device)
            else:
                raise NotImplementedError()

        elif deg == 'motion_deblur':
            from functions.motionblur import Kernel
            if args.operator_imp == 'FFT':
                from functions.fft_operators import Deblurring_fft
            else:
                raise ValueError("set operator_imp = FFT")

        else:
            raise ValueError("degradation type not supported")
        
        args.sigma_y = 2 * args.sigma_y #to account for scaling to [-1,1]
        sigma_y = args.sigma_y
        
        print(f'Start from {args.subset_start}')
        idx_init = args.subset_start
        idx_so_far = args.subset_start
        avg_psnr = 0.0
        avg_lpips = 0.0
        pbar = tqdm.tqdm(val_loader)

        img_ind = -1

        for x_orig, classes in pbar:

            img_ind = img_ind + 1

            if deg == 'motion_deblur':
                # Create different motion for every image
                np.random.seed(seed=img_ind * 10)  # for reproducibility of blur kernel for each image
                kernel = torch.from_numpy(Kernel(size=(61, 61), intensity=0.5).kernelMatrix)
                A_funcs = Deblurring_fft(kernel / kernel.sum(), config.data.channels, self.config.data.image_size, self.device)
                np.random.seed(seed=args.seed) # Back to original seed for reproducibility

            x_orig = x_orig.to(self.device)
            x_orig = data_transform(self.config, x_orig)

            y = A_funcs.A(x_orig)
            
            y = y + args.sigma_y*torch.randn_like(y).cuda()  # added noise to measurement
            
            b, hwc = y.size()
            if 'color' in deg:
                hw = hwc / 1
                h = w = int(hw ** 0.5)
                y = y.reshape((b, 1, h, w))
            elif 'inp' in deg or 'cs' in deg:
                pass
            else:
                hw = hwc / 3
                h = w = int(hw ** 0.5)
                y = y.reshape((b, 3, h, w))
            
            y = y.reshape((b, hwc))

            Apy = A_funcs.A_pinv_add_eta(y, max(1e-4, sigma_y**2 * args.eta_tilde )).view(y.shape[0], config.data.channels, self.config.data.image_size,
                                                self.config.data.image_size)
            

            if args.save_observed_img:
                os.makedirs(os.path.join(self.args.image_folder, "Apy"), exist_ok=True)
                for i in range(len(Apy)):
                    tvu.save_image(
                        inverse_data_transform(config, Apy[i]),
                        os.path.join(self.args.image_folder, f"Apy/Apy_{idx_so_far + i}.png")
                    )
                    tvu.save_image(
                        inverse_data_transform(config, x_orig[i]),
                        os.path.join(self.args.image_folder, f"Apy/orig_{idx_so_far + i}.png")
                    )
                    if 'inp' in deg or 'cs' in deg:
                        pass
                    else:
                        tvu.save_image(
                            inverse_data_transform(config, y[i].reshape((3, h, w))),
                            os.path.join(self.args.image_folder, f"Apy/y_{idx_so_far + i}.png")
                        )

            # initialize x
            x = torch.randn(
                y.shape[0],
                config.data.channels,
                config.data.image_size,
                config.data.image_size,
                device=self.device,
            )

            with torch.no_grad():           
                x, _ = ddpg_diffusion(x, model, self.betas, A_funcs, y, sigma_y, cls_fn=cls_fn, classes=classes, config=config, args=args)
                
                #x, _ = ddpg_diffusion_tom(x, model, self.betas, A_funcs, y, sigma_y, cls_fn=cls_fn, classes=classes, config=config, args=args)


            lpips_final = torch.squeeze(loss_fn_alex(x[0], x_orig.to('cpu'))).detach().numpy()
            avg_lpips += lpips_final

            x = [inverse_data_transform(config, xi) for xi in x]

            for j in range(x[0].size(0)):
                tvu.save_image(
                    x[0][j], os.path.join(self.args.image_folder, f"{idx_so_far + j}_{0}.png")
                )
                orig = inverse_data_transform(config, x_orig[j])
                mse = torch.mean((x[0][j].to(self.device) - orig) ** 2)
                psnr = 10 * torch.log10(1 / mse)
                logger.info("img_ind: %d, PSNR: %.2f, LPIPS: %.4f" % (img_ind, psnr, lpips_final))
                avg_psnr += psnr

            idx_so_far += y.shape[0]

            #pbar.set_description("Avg PSNR: %.2f, Avg LPIPS: %.4f     (** After %d iteration **)" % (avg_psnr / (idx_so_far - idx_init), avg_lpips / (idx_so_far - idx_init), idx_so_far - idx_init))
            logger.info("Avg PSNR: %.2f, Avg LPIPS: %.4f     (** After %d iteration **)" % (avg_psnr / (idx_so_far - idx_init), avg_lpips / (idx_so_far - idx_init), idx_so_far - idx_init))


        avg_psnr = avg_psnr / (idx_so_far - idx_init)
        avg_lpips = avg_lpips / (idx_so_far - idx_init)
        print("Total Average PSNR: %.2f" % avg_psnr)
        print("Total Average LPIPS: %.4f" % avg_lpips)
        print("Number of samples: %d" % (idx_so_far - idx_init))     


