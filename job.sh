python -u train_model.py --model_name unet --in_channels 1 --gpu_bs 1 | tee -a training.log
wait
python -u train_model.py --model_name deeplabv3plus --in_channels 1 --gpu_bs 1 | tee -a training.log
wait
