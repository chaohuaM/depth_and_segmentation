#python -u train_model.py --model_name unet --in_channels 1 --gpu_bs 1 | tee -a training.log
#wait
#python -u train_model.py --model_name deeplabv3plus --in_channels 1 --gpu_bs 1 | tee -a training.log
#wait

#python -u train_model_pl.py --model_name unet --backbone resnet50 --in_channels 3 --gpu_bs 8 --epoch 150 --dataset dataset/rockA+B --data_transform 1 --log_dir ./real-logs | tee -a training.log
#wait
#python -u train_model_pl.py --model_name unet_dual_decoder --backbone resnet50 --in_channels 3 --gpu_bs 8 --epoch 150 --dataset dataset/rockA+B --data_transform 1 --log_dir ./real-logs| tee -a training.log
#wait
#python -u train_model_pl.py --model_name unet_dual_decoder_with_sa --backbone resnet50 --in_channels 3 --gpu_bs 8 --epoch 150 --dataset dataset/rockA+B --data_transform 1 --log_dir ./real-logs | tee -a training.log
#wait
#python -u train_model_pl.py --model_name unet --backbone resnet50 --in_channels 3 --gpu_bs 8 --epoch 150 --dataset dataset/rockA+B --data_transform 1 --log_dir ./real-logs | tee -a training.log
#wait
#python -u train_model_pl.py --model_name unet_dual_decoder --backbone resnet50 --in_channels 3 --gpu_bs 8 --epoch 150 --dataset dataset/rockA+B --data_transform 1 --log_dir ./real-logs| tee -a training.log
#wait
#python -u train_model_pl.py --model_name unet_dual_decoder_with_sa --backbone resnet50 --in_channels 3 --gpu_bs 8 --epoch 150 --dataset dataset/rockA+B --data_transform 1 --log_dir ./real-logs | tee -a training.log
#wait


#python -u train_model_pl.py --model_name unet_dual_decoder_with_sa --in_channels 3 --gpu_bs 8 --epoch 150 --depth_loss_fn berhu_loss --log_dir ./51-logs/ | tee -a training.log
#wait
#python -u train_model_pl.py --model_name unet_dual_decoder_with_sa --in_channels 3 --gpu_bs 8 --epoch 150 --depth_loss_fn berhu_loss --use_depth_mask 0 --log_dir ./51-logs/ | tee -a training.log
#wait
#python -u train_model_pl.py --model_name unet --in_channels 3 --gpu_bs 8 --epoch 150 --log_dir ./51-logs/ | tee -a training.log
#wait
#python -u train_model_pl.py --model_name unet_dual_decoder_with_sa --in_channels 3 --gpu_bs 8 --epoch 150 --depth_loss_fn berhu_loss --log_dir ./51-logs/ | tee -a training.log
#wait
#python -u train_model_pl.py --model_name unet_dual_decoder --in_channels 3 --gpu_bs 8 --epoch 150 --depth_loss_fn berhu_loss --log_dir ./51-logs/ | tee -a training.log
#wait
#python -u train_model_pl.py --model_name unet_dual_decoder_with_sa --in_channels 3 --gpu_bs 8 --epoch 150 --depth_loss_fn ssi_loss --log_dir ./51-logs/ --data_transform 1 | tee -a training.log
#wait

python -u train_model_pl.py --model_name unet --in_channels 3 --gpu_bs 8 --epoch 150 --log_dir ./real-logs/ --gpus 1 | tee -a training.log
wait
python -u train_model_pl.py --model_name unet_dual_decoder --in_channels 3 --gpu_bs 8 --epoch 150 --depth_loss_fn berhu_loss --log_dir ./real-logs/ --gpus 1 | tee -a training.log
wait
python -u train_model_pl.py --model_name unet_dual_decoder --in_channels 3 --gpu_bs 8 --epoch 150 --depth_loss_fn ssi_loss --log_dir ./real-logs/ --gpus 1 | tee -a training.log
wait
python -u train_model_pl.py --model_name unet_dual_decoder_with_sa --in_channels 3 --gpu_bs 8 --epoch 150 --depth_loss_fn berhu_loss --log_dir ./real-logs/ --gpus 1 | tee -a training.log
wait
python -u train_model_pl.py --model_name unet_dual_decoder_with_sa --in_channels 3 --gpu_bs 8 --epoch 150 --depth_loss_fn ssi_loss --log_dir ./real-logs/ --gpus 1 | tee -a training.log
wait