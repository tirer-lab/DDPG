## Experiments on CelebA ##

# Noiseless tasks

# Super-Resolution Bicubic
python main.py --config celeba_hq.yml --path_y celeba_hq --deg sr_bicubic --sigma_y 0 \
-i DDPG_celeba_sr_bicubic_sigma_y_0 --inject_noise 1 --zeta 0.7 --step_size_mode 0 \
--deg_scale 4 --operator_imp SVD

# Gaussian Deblurring

# Use SVD to reproduce the paper's results.
python main.py --config celeba_hq.yml --path_y celeba_hq --deg deblur_gauss --sigma_y 0 \
-i DDPG_celeba_deblur_gauss_sigma_y_0 --inject_noise 1 --zeta 1.0 --step_size_mode 0 \
--operator_imp FFT


# 0.01 Noise tasks

# Super-Resolution Bicubic
python main.py --config celeba_hq.yml --path_y celeba_hq --deg sr_bicubic --sigma_y 0.01 \
-i DDPG_celeba_sr_bicubic_sigma_y_0.01 --inject_noise 1 --gamma 300 --zeta 1.0 --eta_tilde 1.0 \
--step_size_mode 0 --deg_scale 4 --operator_imp SVD

# Gaussian Deblurring

# Use SVD to reproduce the paper's results.
python main.py --config celeba_hq.yml --path_y celeba_hq --deg deblur_gauss --sigma_y 0.01 \
-i DDPG_celeba_deblur_gauss_sigma_y_0.01 --inject_noise 1 --gamma 11 --zeta 0.6 --eta_tilde 1.0 \
--step_size_mode 0 --operator_imp FFT

# Motion Deblurring
python main.py --config celeba_hq.yml --path_y celeba_hq --deg motion_deblur --sigma_y 0.01 \
-i DDPG_celeba_motion_deblur_sigma_y_0.01 --inject_noise 1 --gamma 50 --zeta 0.5 --eta_tilde 6.0 \
--step_size_mode 0 --operator_imp FFT

# 0.05 Noise tasks

# Super-Resolution Bicubic
python main.py --config celeba_hq.yml --path_y celeba_hq --deg sr_bicubic --sigma_y 0.05 \
-i DDPG_celeba_sr_bicubic_sigma_y_0.05 --inject_noise 1 --gamma 10 --zeta 0.8 --eta_tilde 0.3 \
--step_size_mode 1 --deg_scale 4 --operator_imp SVD

# Gaussian Deblurring

# Use SVD to reproduce the paper's results.
python main.py --config celeba_hq.yml --path_y celeba_hq --deg deblur_gauss --sigma_y 0.05 \
-i DDPG_celeba_deblur_gauss_sigma_y_0.05 --inject_noise 1 --gamma 8 --zeta 0.5 --eta_tilde 0.7 \
--step_size_mode 1 --operator_imp FFT


# Motion Deblurring
python main.py --config celeba_hq.yml --path_y celeba_hq --deg motion_deblur --sigma_y 0.05 \
-i DDPG_celeba_motion_deblur_sigma_y_0.05 --inject_noise 1 --gamma 5 --zeta 0.6 --eta_tilde 0.6 \
--step_size_mode 1 --operator_imp FFT

# 0.1 Noise tasks

# Gaussian Deblurring

# Use SVD to reproduce the paper's results.
python main.py --config celeba_hq.yml --path_y celeba_hq --deg deblur_gauss --sigma_y 0.1 \
-i DDPG_celeba_deblur_gauss_sigma_y_0.1 --inject_noise 1 --gamma 5 --zeta 0.6 --eta_tilde 0.7 \
--step_size_mode 1 --operator_imp FFT


# Motion Deblurring
python main.py --config celeba_hq.yml --path_y celeba_hq --deg motion_deblur --sigma_y 0.1 \
-i DDPG_celeba_motion_deblur_sigma_y_0.1 --inject_noise 1 --gamma 5 --zeta 0.6 --eta_tilde 0.6 \
--step_size_mode 1 --operator_imp FFT

## Experiments on ImageNet ##

# Noiseless tasks

# Super-Resolution Bicubic
python main.py --config imagenet_256.yml --path_y imagenet --deg sr_bicubic --sigma_y 0 \
-i DDPG_imagenet_sr_bicubic_sigma_y_0 --inject_noise 1 --zeta 0.7 --step_size_mode 0 \
--deg_scale 4 --operator_imp SVD

# Gaussian Deblurring

# Use SVD to reproduce the paper's results.
python main.py --config imagenet_256.yml --path_y imagenet --deg deblur_gauss --sigma_y 0 \
-i DDPG_imagenet_deblur_gauss_sigma_y_0 --inject_noise 1 --zeta 1.0 --step_size_mode 0 \
--operator_imp FFT


# 0.05 Noise tasks

# Super-Resolution Bicubic
python main.py --config imagenet_256.yml --path_y imagenet --deg sr_bicubic --sigma_y 0.05 \
-i DDPG_imagenet_sr_bicubic_sigma_y_0.05 --inject_noise 1 --gamma 6 --zeta 1.0 --eta_tilde 0.3 \
--step_size_mode 1 --deg_scale 4 --operator_imp SVD

# Gaussian Deblurring

# Use SVD to reproduce the paper's results.
python main.py --config imagenet_256.yml --path_y imagenet --deg deblur_gauss --sigma_y 0.05 \
-i DDPG_imagenet_deblur_gauss_sigma_y_0.05 --inject_noise 1 --gamma 10 --zeta 0.4 --eta_tilde 0.7 \
--step_size_mode 1 --operator_imp FFT


# Motion Deblurring
python main.py --config imagenet_256.yml --path_y imagenet --deg motion_deblur --sigma_y 0.05 \
-i DDPG_imagenet_motion_deblur_sigma_y_0.05 --inject_noise 1 --gamma 6 --zeta 0.6 --eta_tilde 0.7 \
--step_size_mode 1 --operator_imp FFT

# 0.1 Noise tasks

# Motion Deblurring
python main.py --config imagenet_256.yml --path_y imagenet --deg motion_deblur --sigma_y 0.1 \
-i DDPG_imagenet_motion_deblur_sigma_y_0.1 --inject_noise 1 --gamma 3 --zeta 0.6 --eta_tilde 0.4 \
--step_size_mode 1 --operator_imp FFT
