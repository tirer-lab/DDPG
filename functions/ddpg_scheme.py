import torch
from tqdm import tqdm
import torchvision.utils as tvu
import torchvision
import os
import numpy as np
from datasets import inverse_data_transform, data_transform

class_num = 951


def compute_alpha(beta, t):
    beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
    a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
    return a


def ddpg_diffusion(x, model, b, A_funcs, y, sigma_y, cls_fn=None, classes=None, config=None, args=None):
    with torch.no_grad():

        # setup iteration variables
        skip = config.diffusion.num_diffusion_timesteps//config.sampling.T_sampling
        n = x.size(0)
        x0_preds = []
        xs = [x]

        # generate time schedule
        times = get_schedule_jump(config.sampling.T_sampling, 1, 1)
        time_pairs = list(zip(times[:-1], times[1:]))        
        
        # reverse diffusion sampling
        for i, j in tqdm(time_pairs):

            i, j = i*skip, j*skip
            if j<0: j=-1 

            if j < i: # normal sampling 

                t = (torch.ones(n) * i).to(x.device)
                next_t = (torch.ones(n) * j).to(x.device)
                at = compute_alpha(b, t.long())
                at_next = compute_alpha(b, next_t.long())
                xt = xs[-1].to('cuda')
                if cls_fn == None:
                    et = model(xt, t)
                else:
                    classes = torch.ones(xt.size(0), dtype=torch.long, device=torch.device("cuda"))*class_num
                    et = model(xt, t, classes)
                    et = et[:, :3]
                    et = et - (1 - at).sqrt()[0, 0, 0, 0] * cls_fn(x, t, classes)

                if et.size(1) == 6:
                    et = et[:, :3]

                # estimate x0
                x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()

                if sigma_y==0.:
                    delta_t = 0
                    weight_noise_t = 1
                else:
                    delta_t = (at_next) ** args.gamma
                    weight_noise_t = delta_t


                eta_reg = max(1e-4, sigma_y**2 * args.eta_tilde )
                if args.eta_tilde < 0:
                    eta_reg = 1e-4 + args.xi * (sigma_y*255.0)**2

                scale_gLS = args.scale_ls  #e.g. <= 1/A_funcs.singulars().max()**2 

                guidance_BP = A_funcs.A_pinv_add_eta(A_funcs.A(x0_t.reshape(x0_t.size(0), -1)) - y.reshape(y.size(0), -1), eta_reg).reshape(*x0_t.size())
                guidance_LS = A_funcs.At(A_funcs.A(x0_t.reshape(x0_t.size(0), -1)) - y.reshape(y.size(0), -1)).reshape(*x0_t.size())

                if args.step_size_mode==0:
                    step_size_LS = 1
                    step_size_BP = 1
                    step_size = 1
                elif args.step_size_mode==1:
                    step_size_LS = 1
                    step_size_BP = 1
                    step_size = (1 - at_next)/(1 - at)
                elif args.step_size_mode==2:
                    step_size_LS = (1 - at_next)/(1 - at)
                    step_size_BP = 1
                    step_size = 1
                else:
                    assert 1, "unsupported step-size mode"

                # data fidelity guidance
                xt_next_tilde = x0_t - step_size * ( step_size_BP * (1-delta_t) * guidance_BP + step_size_LS * delta_t * scale_gLS * guidance_LS )

                # compute effective noise
                et_hat = ( xt - at.sqrt() * xt_next_tilde ) / (1 - at).sqrt()

                c1 = 0
                c2 = 0
                if args.inject_noise:
                    zeta = args.zeta
                    c1 = (1 - at_next).sqrt() * np.sqrt(zeta)
                    c2 = (1 - at_next).sqrt() * np.sqrt(1-zeta)  * weight_noise_t

                xt_next = at_next.sqrt() * xt_next_tilde + c1 * torch.randn_like(x0_t) + c2 * et_hat


                x0_preds.append(x0_t.to('cpu'))
                xs.append(xt_next.to('cpu'))

            else: 
                assert 1, "Unexpected case"
        
        if sigma_y != 0.:  # if there is noise, take the denoised result
            xs.append(x0_t.to('cpu'))

    return [xs[-1]], [x0_preds[-1]]


# code from RePaint
def get_schedule_jump(T_sampling, travel_length, travel_repeat):

    jumps = {}
    for j in range(0, T_sampling - travel_length, travel_length):
        jumps[j] = travel_repeat - 1

    t = T_sampling
    ts = []

    while t >= 1:
        t = t-1
        ts.append(t)

        if jumps.get(t, 0) > 0:
            jumps[t] = jumps[t] - 1
            for _ in range(travel_length):
                t = t + 1
                ts.append(t)

    ts.append(-1)

    _check_times(ts, -1, T_sampling)

    return ts

def _check_times(times, t_0, T_sampling):
    # Check end
    assert times[0] > times[1], (times[0], times[1])

    # Check beginning
    assert times[-1] == -1, times[-1]

    # Steplength = 1
    for t_last, t_cur in zip(times[:-1], times[1:]):
        assert abs(t_last - t_cur) == 1, (t_last, t_cur)

    # Value range
    for t in times:
        assert t >= t_0, (t, t_0)
        assert t <= T_sampling, (t, T_sampling)


