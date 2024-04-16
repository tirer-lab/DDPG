## Experiments on CelebA ##

# Noiseless tasks

# Super-Resolution Bicubic
python main.py --config celeba_hq.yml --path_y celeba_hq --deg sr_bicubic --sigma_y 0 \
-i IDPG_celeba_sr_bicubic_sigma_y_0 --inject_noise 0 --step_size_mode 0 --deg_scale 4 --operator_imp SVD

# Gaussian Deblurring

# Use SVD to reproduce the paper's results.
python main.py --config celeba_hq.yml --path_y celeba_hq --deg deblur_gauss --sigma_y 0 \
-i IDPG_celeba_deblur_gauss_sigma_y_0 --inject_noise 0 --step_size_mode 0 --operator_imp FFT

# 0.01 Noise tasks

# Super-Resolution Bicubic
python main.py --config celeba_hq.yml --path_y celeba_hq --deg sr_bicubic --sigma_y 0.01 \
-i IDPG_celeba_sr_bicubic_sigma_y_0.01 --inject_noise 0 --gamma 300 --eta_tilde 0 --step_size_mode 0 \
--deg_scale 4 --operator_imp SVD

# Gaussian Deblurring

# Use SVD to reproduce the paper's results.
python main.py --config celeba_hq.yml --path_y celeba_hq --deg deblur_gauss --sigma_y 0.01 \
-i IDPG_celeba_deblur_gauss_sigma_y_0.01 --inject_noise 0 --gamma 100 --eta_tilde 7.0 --step_size_mode 0 \
--operator_imp FFT

# Motion Deblurring
python main.py --config celeba_hq.yml --path_y celeba_hq --deg motion_deblur --sigma_y 0.01 \
-i IDPG_celeba_motion_deblur_sigma_y_0.01 --inject_noise 0 --gamma 90 --eta_tilde 7.0 --step_size_mode 0 \
--operator_imp FFT

# 0.05 Noise tasks

# Super-Resolution Bicubic
python main.py --config celeba_hq.yml --path_y celeba_hq --deg sr_bicubic --sigma_y 0.05 \
-i IDPG_celeba_sr_bicubic_sigma_y_0.05 --inject_noise 0 --gamma 16 --eta_tilde 0.2 --step_size_mode 0 \
--deg_scale 4 --operator_imp SVD

# Gaussian Deblurring

# Use SVD to reproduce the paper's results.
python main.py --config celeba_hq.yml --path_y celeba_hq --deg deblur_gauss --sigma_y 0.05 \
-i IDPG_celeba_deblur_gauss_sigma_y_0.05 --inject_noise 0 --gamma 8 --eta_tilde 0.6 --step_size_mode 0 \
--operator_imp FFT

# Motion Deblurring
python main.py --config celeba_hq.yml --path_y celeba_hq --deg motion_deblur --sigma_y 0.05 \
-i IDPG_celeba_motion_deblur_sigma_y_0.05 --inject_noise 0 --gamma 12 --eta_tilde 0.9 --step_size_mode 0 \
--operator_imp FFT

# 0.1 Noise tasks

# Gaussian Deblurring

# Use SVD to reproduce the paper's results.
python main.py --config celeba_hq.yml --path_y celeba_hq --deg deblur_gauss --sigma_y 0.1 \
-i IDPG_celeba_deblur_gauss_sigma_y_0.1 --inject_noise 0 --gamma 6 --eta_tilde 0.6 --step_size_mode 0 \
--operator_imp FFT

# Motion Deblurring
python main.py --config celeba_hq.yml --path_y celeba_hq --deg motion_deblur --sigma_y 0.1 \
-i IDPG_celeba_motion_deblur_sigma_y_0.1 --inject_noise 0 --gamma 14 --eta_tilde 1.0 --step_size_mode 0 \
--operator_imp FFT

## Experiments on ImageNet ##

# Noiseless tasks

# Super-Resolution Bicubic
python main.py --config imagenet_256.yml --path_y imagenet --deg sr_bicubic --sigma_y 0 \
-i IDPG_imagenet_sr_bicubic_sigma_y_0 --inject_noise 0 --step_size_mode 0 --deg_scale 4 --operator_imp SVD

# Gaussian Deblurring

# Use SVD to reproduce the paper's results.
python main.py --config imagenet_256.yml --path_y imagenet --deg deblur_gauss --sigma_y 0 \
-i IDPG_imagenet_deblur_gauss_sigma_y_0 --inject_noise 0 --step_size_mode 0 --operator_imp FFT

# 0.05 Noise tasks

# Super-Resolution Bicubic
python main.py --config imagenet_256.yml --path_y imagenet --deg sr_bicubic --sigma_y 0.05 \
-i IDPG_imagenet_sr_bicubic_sigma_y_0.05 --inject_noise 0 --gamma 30 --eta_tilde 0.2 \
--step_size_mode 0 --deg_scale 4 --operator_imp SVD

# Gaussian Deblurring

# Use SVD to reproduce the paper's results.
python main.py --config imagenet_256.yml --path_y imagenet --deg deblur_gauss --sigma_y 0.05 \
-i IDPG_imagenet_deblur_gauss_sigma_y_0.05 --inject_noise 0 --gamma 11 --eta_tilde 0.6 --step_size_mode 0 \
--operator_imp FFT

# Motion Deblurring
python main.py --config imagenet_256.yml --path_y imagenet --deg motion_deblur --sigma_y 0.05 \
-i IDPG_imagenet_motion_deblur_sigma_y_0.05 --inject_noise 0 --gamma 14 --eta_tilde 0.8 --step_size_mode 0 \
--operator_imp FFT

# 0.1 Noise tasks
# Motion Deblurring
python main.py --config imagenet_256.yml --path_y imagenet --deg motion_deblur --sigma_y 0.1 \
-i IDPG_imagenet_motion_deblur_sigma_y_0.1 --inject_noise 0 --gamma 11 --eta_tilde 0.6 --step_size_mode 0 \
--operator_imp FFT