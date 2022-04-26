#python -u train_model.py --model_name unet --in_channels 1 --gpu_bs 1 | tee -a training.log
#wait
#python -u train_model.py --model_name deeplabv3plus --in_channels 1 --gpu_bs 1 | tee -a training.log
#wait

python -u train_model_pl.py --model_name unet_dual_decoder_with_sa --in_channels 3 --gpu_bs 1 | tee -a training.log
wait