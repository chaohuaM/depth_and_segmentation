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

#python -u train_model_pl.py --model_name unet --in_channels 3 --gpu_bs 8 --epoch 150 --log_dir ./real-logs/ --gpus 1 | tee -a training.log
#wait
#python -u train_model_pl.py --model_name unet_dual_decoder --in_channels 3 --gpu_bs 8 --epoch 150 --depth_loss_fn berhu_loss --log_dir ./real-logs/ --gpus 1 | tee -a training.log
#wait
#python -u train_model_pl.py --model_name unet_dual_decoder --in_channels 3 --gpu_bs 8 --epoch 150 --depth_loss_fn ssi_loss --log_dir ./real-logs/ --gpus 1 | tee -a training.log
#wait
#python -u train_model_pl.py --model_name unet_dual_decoder_with_sa --in_channels 3 --gpu_bs 8 --epoch 150 --depth_loss_fn berhu_loss --log_dir ./real-logs/ --gpus 1 | tee -a training.log
#wait
#python -u train_model_pl.py --model_name unet_dual_decoder_with_sa --in_channels 3 --gpu_bs 8 --epoch 150 --depth_loss_fn ssi_loss --log_dir ./real-logs/ --gpus 1 | tee -a training.log
#wait
#python -u train_model_pl.py --model_name unet --in_channels 3 --gpu_bs 8 --epoch 100 --log_dir ./525-logs/ --gpus 0 | tee -a training.log
#wait
#python -u train_model_pl.py --model_name unet_dual_decoder --in_channels 3 --gpu_bs 8 --epoch 100 --depth_loss_fn berhu_loss --use_depth_mask 0 --log_dir ./525-logs/ --gpus 0 | tee -a training.log
#wait
#python -u train_model_pl.py --model_name unet_dual_decoder --in_channels 3 --gpu_bs 8 --epoch 100 --depth_loss_fn ssi_loss --use_depth_mask 0 --log_dir ./525-logs/ --gpus 0 | tee -a training.log
#wait
#python -u train_model_pl.py --model_name unet_dual_decoder_with_sa --in_channels 3 --gpu_bs 8 --epoch 100 --depth_loss_fn berhu_loss --use_depth_mask 0 --log_dir ./525-logs/ --gpus 0 | tee -a training.log
#wait
#python -u train_model_pl.py --model_name unet_dual_decoder_with_sa --in_channels 3 --gpu_bs 8 --epoch 100 --depth_loss_fn ssi_loss --use_depth_mask 0 --log_dir ./525-logs/ --gpus 0 | tee -a training.log
#wait
#python -u train_model_pl.py --model_name unet_dual_decoder_with_sa --lr 0.1 --in_channels 3 --gpu_bs 8 --epoch 100 --depth_loss_fn berhu_loss --use_depth_mask 0 --log_dir ./test-logs/ --gpus 1 | tee -a training.log
#wait
#python -u train_model_pl.py --model_name unet_dual_decoder_with_sa --lr 0.1 --in_channels 3 --gpu_bs 8 --epoch 100 --depth_loss_fn ssi_loss --use_depth_mask 1 --log_dir ./525-logs/ --gpus 1 | tee -a training.log
#wait
##python -u train_model_pl.py --model_name unet_dual_decoder_with_sa --lr 0.00001 --in_channels 3 --gpu_bs 8 --epoch 150 --depth_loss_fn ssi_loss --use_depth_mask 1 --log_dir ./525-logs/ --gpus 0 | tee -a training.log
##wait
#python -u train_model_pl.py --model_name unet_dual_decoder_with_sa --lr 0.1 --in_channels 3 --gpu_bs 8 --epoch 200 --depth_loss_fn ssi_loss --use_depth_mask 0 --log_dir ./525-logs/ --gpus 1 | tee -a training.log
#wait
#python -u train_model_pl.py --model_name unet --lr 0.00001 --in_channels 3 --gpu_bs 8 --epoch 100  --log_dir ./602-logs/ --gpus 1 | tee -a training.log
#wait
#python -u train_model_pl.py --model_name unet_dual_decoder --lr 0.00001 --in_channels 3 --gpu_bs 8 --epoch 100 --depth_loss_fn berhu_loss --use_depth_mask 0 --log_dir ./602-logs/ --gpus 1 | tee -a training.log
#wait
#python -u train_model_pl.py --model_name unet_dual_decoder --lr 0.00001 --in_channels 3 --gpu_bs 8 --epoch 100 --depth_loss_fn ssi_loss --use_depth_mask 0 --log_dir ./602-logs/ --gpus 1 | tee -a training.log
#wait
#python -u train_model_pl.py --model_name unet_dual_decoder_with_sa --lr 0.00001 --in_channels 3 --gpu_bs 8 --epoch 100 --depth_loss_fn berhu_loss --use_depth_mask 0 --log_dir ./602-logs/ --gpus 1 | tee -a training.log
#wait
#python -u train_model_pl.py --model_name unet_dual_decoder_with_sa --lr 0.00001 --in_channels 3 --gpu_bs 8 --epoch 100 --depth_loss_fn ssi_loss --use_depth_mask 0 --log_dir ./602-logs/ --gpus 1 | tee -a training.log
#wait
#python -u train_model_pl.py --model_name unet_dual_decoder --lr 0.00001 --in_channels 3 --gpu_bs 8 --epoch 100 --depth_loss_fn berhu_loss --use_depth_mask 1 --log_dir ./602-logs/ --gpus 1 | tee -a training.log
#wait
#python -u train_model_pl.py --model_name unet_dual_decoder --lr 0.00001 --in_channels 3 --gpu_bs 8 --epoch 100 --depth_loss_fn ssi_loss --use_depth_mask 1 --log_dir ./602-logs/ --gpus 1 | tee -a training.log
#wait
#python -u train_model_pl.py --model_name unet_dual_decoder_with_sa --lr 0.00001 --in_channels 3 --gpu_bs 8 --epoch 100 --depth_loss_fn berhu_loss --use_depth_mask 1 --log_dir ./602-logs/ --gpus 1 | tee -a training.log
#wait
#python -u train_model_pl.py --model_name unet_dual_decoder_with_sa --lr 0.00001 --in_channels 3 --gpu_bs 8 --epoch 100 --depth_loss_fn ssi_loss --use_depth_mask 1 --log_dir ./602-logs/ --gpus 1 | tee -a training.log
#wait

#python -u train_model_pl.py --model_name unet_dual_decoder_with_sa --lr 0.1 --optimizer Adam --in_channels 3 --gpu_bs 8 --epoch 100 --depth_loss_fn ssi_loss --use_depth_mask 0 --log_dir ./test-logs/ --gpus 1 | tee -a training.log
#wait
#python -u train_model_pl.py --model_name unet_dual_decoder_with_sa --lr 0.1 --optimizer SGD --in_channels 3 --gpu_bs 8 --epoch 100 --depth_loss_fn ssi_loss --use_depth_mask 0 --log_dir ./test-logs/ --gpus 1 | tee -a training.log
#wait

#python -u train_model_pl.py --model_name unet_dual_decoder_with_sa --lr 0.0001 --optimizer Adam --in_channels 3 --gpu_bs 8 --epoch 100 --depth_loss_fn ssi_loss --use_depth_mask 0 --log_dir ./test-logs/ --gpus 1 | tee -a training.log
#wait
#python -u train_model_pl.py --model_name unet_dual_decoder --lr 0.0001 --optimizer Adam --in_channels 3 --gpu_bs 8 --epoch 100 --depth_loss_fn ssi_loss --use_depth_mask 0 --log_dir ./test-logs/ --gpus 1 | tee -a training.log
#wait
#python -u train_model_pl.py --model_name unet --lr 0.0001 --optimizer Adam --in_channels 3 --gpu_bs 8 --epoch 100 --depth_loss_fn ssi_loss --use_depth_mask 0 --log_dir ./test-logs/ --gpus 1 | tee -a training.log
#wait
#python -u train_model_pl.py --model_name unet --lr 0.001 --optimizer Adam --in_channels 3 --gpu_bs 8 --epoch 100 --depth_loss_fn ssi_loss --use_depth_mask 0 --log_dir ./test-logs/ --gpus 1 | tee -a training.log
#wait
#python -u train_model_pl.py --model_name unet --lr 0.01 --optimizer Adam --in_channels 3 --gpu_bs 8 --epoch 100 --depth_loss_fn ssi_loss --use_depth_mask 0 --log_dir ./test-logs/ --gpus 1 | tee -a training.log
#wait
#python -u train_model_pl.py --model_name unet_dual_decoder --lr 0.0001 --optimizer Adam --in_channels 3 --gpu_bs 8 --epoch 100 --depth_loss_fn ssi_loss --use_depth_mask 0 --log_dir ./test-logs/ --gpus 1 | tee -a training.log
#wait
#python -u train_model_pl.py --model_name unet_dual_decoder --lr 0.001 --optimizer Adam --in_channels 3 --gpu_bs 8 --epoch 100 --depth_loss_fn ssi_loss --use_depth_mask 0 --log_dir ./test-logs/ --gpus 1 | tee -a training.log
#wait
#python -u train_model_pl.py --model_name unet_dual_decoder --lr 0.01 --optimizer Adam --in_channels 3 --gpu_bs 8 --epoch 100 --depth_loss_fn ssi_loss --use_depth_mask 0 --log_dir ./test-logs/ --gpus 1 | tee -a training.log
#wait
#python -u train_model_pl.py --model_name unet_dual_decoder_with_sa --lr 0.0001 --optimizer Adam --in_channels 3 --gpu_bs 8 --epoch 100 --depth_loss_fn ssi_loss --use_depth_mask 0 --log_dir ./test-logs/ --gpus 1 | tee -a training.log
#wait
# 上次跑到这里
#python -u train_model_pl.py --model_name unet_dual_decoder_with_sa --lr 0.001 --optimizer Adam --in_channels 3 --gpu_bs 8 --epoch 100 --depth_loss_fn ssi_loss --use_depth_mask 0 --log_dir ./test-logs/ --gpus 1 | tee -a training.log
#wait
#python -u train_model_pl.py --model_name unet_dual_decoder_with_sa --lr 0.01 --optimizer Adam --in_channels 3 --gpu_bs 8 --epoch 100 --depth_loss_fn ssi_loss --use_depth_mask 0 --log_dir ./test-logs/ --gpus 1 | tee -a training.log
#wait
#python -u train_model_pl.py --model_name unet_dual_decoder_with_sa --lr 0.00001 --optimizer Adam --in_channels 3 --gpu_bs 8 --epoch 100 --depth_loss_fn ssi_loss --use_depth_mask 0 --log_dir ./test-logs/ --gpus 1 | tee -a training.log
#wait

#python -u train_model_pl.py --model_name unet --lr 0.0001 --optimizer Adam --in_channels 1 --batch_size 8 --gpu_bs 8 --epoch 150 --depth_loss_fn ssi_loss --use_depth_mask 0 --log_dir ./MER-logs/ --gpus 1 --dataset_path ./dataset/MER | tee -a training.log
#wait
#python -u train_model_pl.py --model_name unet --lr 0.0001 --optimizer Adam --in_channels 1 --batch_size 8 --gpu_bs 8 --epoch 100 --depth_loss_fn ssi_loss --use_depth_mask 0 --log_dir ./MER-logs/ --gpus 1 --dataset_path ./dataset/MER | tee -a training.log
#wait

#python -u train_model_pl.py --model_name unet --lr 0.1 --optimizer Adam --in_channels 1 --batch_size 8 --gpu_bs 8 --epoch 150 --depth_loss_fn berhu_loss --use_depth_mask 0 --log_dir ./MER-logs/ --gpus 1 --dataset_path ./dataset/MER | tee -a training.log
#wait
#python -u train_model_pl.py --model_name unet_dual_decoder --lr 0.1 --optimizer Adam --in_channels 1 --batch_size 8 --gpu_bs 8 --epoch 150 --depth_loss_fn ssi_loss --use_depth_mask 0 --log_dir ./MER-logs/ --gpus 1 --dataset_path ./dataset/MER | tee -a training.log
#wait
#python -u train_model_pl.py --model_name unet_dual_decoder_with_sa --lr 0.1 --optimizer Adam --in_channels 1 --batch_size 8 --gpu_bs 8 --epoch 150 --depth_loss_fn ssi_loss --use_depth_mask 0 --log_dir ./MER-logs/ --gpus 1 --dataset_path ./dataset/MER | tee -a training.log
#wait
#python -u train_model_pl.py --model_name unet_dual_decoder --lr 0.1 --optimizer Adam --in_channels 1 --batch_size 8 --gpu_bs 8 --epoch 150 --depth_loss_fn berhu_loss --use_depth_mask 0 --log_dir ./MER-logs/ --gpus 1 --dataset_path ./dataset/MER | tee -a training.log
#wait
#python -u train_model_pl.py --model_name unet_dual_decoder_with_sa --lr 0.1 --optimizer Adam --in_channels 1 --batch_size 8 --gpu_bs 8 --epoch 150 --depth_loss_fn berhu_loss --use_depth_mask 0 --log_dir ./MER-logs/ --gpus 1 --dataset_path ./dataset/MER | tee -a training.log
#wait
#python -u train_model_pl.py --model_name unet --lr 0.001 --optimizer Adam --in_channels 1 --batch_size 8 --gpu_bs 8 --epoch 150 --depth_loss_fn berhu_loss --use_depth_mask 0 --log_dir ./MER-logs/ --gpus 1 --dataset_path ./dataset/MER | tee -a training.log
#wait
#python -u train_model_pl.py --model_name unet_dual_decoder --lr 0.01 --optimizer Adam --in_channels 1 --batch_size 8 --gpu_bs 8 --epoch 150 --depth_loss_fn ssi_loss --use_depth_mask 0 --log_dir ./MER-logs/ --gpus 1 --dataset_path ./dataset/MER | tee -a training.log
#wait
#python -u train_model_pl.py --model_name unet_dual_decoder_with_sa --lr 0.01 --optimizer Adam --in_channels 1 --batch_size 8 --gpu_bs 8 --epoch 150 --depth_loss_fn ssi_loss --use_depth_mask 0 --log_dir ./MER-logs/ --gpus 1 --dataset_path ./dataset/MER | tee -a training.log
#wait
#python -u train_model_pl.py --model_name unet_dual_decoder --lr 0.01 --optimizer Adam --in_channels 1 --batch_size 8 --gpu_bs 8 --epoch 150 --depth_loss_fn berhu_loss --use_depth_mask 0 --log_dir ./MER-logs/ --gpus 1 --dataset_path ./dataset/MER | tee -a training.log
#wait
#python -u train_model_pl.py --model_name unet_dual_decoder_with_sa --lr 0.01 --optimizer Adam --in_channels 1 --batch_size 8 --gpu_bs 8 --epoch 150 --depth_loss_fn berhu_loss --use_depth_mask 0 --log_dir ./MER-logs/ --gpus 1 --dataset_path ./dataset/MER | tee -a training.log
#wait

#python -u train_model_pl.py --model_name unet --lr 0.001 --optimizer Adam --in_channels 1 --batch_size 8 --gpu_bs 8 --epoch 150 --depth_loss_fn berhu_loss --use_depth_mask 0 --log_dir ./MER-logs/ --gpus 1 --dataset_path ./dataset/MER | tee -a training.log
#wait
#python -u train_model_pl.py --model_name unet_dual_decoder --lr 0.001 --optimizer Adam --in_channels 1 --batch_size 4 --gpu_bs 8 --epoch 150 --depth_loss_fn ssi_loss --use_depth_mask 0 --log_dir ./MER-logs/ --gpus 1 --dataset_path ./dataset/MER | tee -a training.log
#wait
#python -u train_model_pl.py --model_name unet --lr 0.01 --optimizer Adam --in_channels 3 --batch_size 8 --gpu_bs 8 --epoch 100  --use_depth_mask 0 --log_dir ./MSL-logs/ --gpus 1 --dataset_path ./dataset/MSL | tee -a training.log
#wait
#python -u train_model_pl.py --model_name unet_dual_decoder --lr 0.01 --optimizer Adam --in_channels 3 --batch_size 8 --gpu_bs 8 --epoch 100 --depth_loss_fn ssi_loss --use_depth_mask 0 --log_dir ./MSL-logs/ --gpus 1 --dataset_path ./dataset/MSL | tee -a training.log
#wait
#python -u train_model_pl.py --model_name unet_dual_decoder_with_sa --lr 0.01 --optimizer Adam --in_channels 3 --batch_size 8 --gpu_bs 8 --epoch 100 --depth_loss_fn ssi_loss --use_depth_mask 0 --log_dir ./MSL-logs/ --gpus 1 --dataset_path ./dataset/MSL | tee -a training.log
#wait
#
#python -u train_model_pl.py --model_name unet --lr 0.0001 --optimizer Adam --in_channels 3 --batch_size 8 --gpu_bs 8 --epoch 100  --use_depth_mask 0 --log_dir ./MSL-aug-logs/ --gpus 0 --dataset_path ./dataset/MSL/aug_data | tee -a training.log
#wait
#python -u train_model_pl.py --model_name unet_dual_decoder --lr 0.0001 --optimizer Adam --in_channels 3 --batch_size 8 --gpu_bs 8 --epoch 100 --depth_loss_fn ssi_loss --use_depth_mask 0 --log_dir ./MSL-aug-logs/ --gpus 0 --dataset_path ./dataset/MSL/aug_data | tee -a training.log
#wait
#python -u train_model_pl.py --model_name unet_dual_decoder_with_sa --lr 0.0001 --optimizer Adam --in_channels 3 --batch_size 8 --gpu_bs 8 --epoch 100 --depth_loss_fn ssi_loss --use_depth_mask 0 --log_dir ./MSL-aug-logs/ --gpus 0 --dataset_path ./dataset/MSL/aug_data | tee -a training.log
#wait
#python -u train_model_pl.py --model_name unet --lr 0.00001 --optimizer Adam --in_channels 3 --batch_size 8 --gpu_bs 8 --epoch 100  --use_depth_mask 0 --log_dir ./MSL-aug-logs/ --gpus 0 --dataset_path ./dataset/MSL/aug_data | tee -a training.log
#wait
python -u train_model_pl.py --model_name unet_dual_decoder --lr 0.001 --optimizer Adam --in_channels 3 --batch_size 8 --gpu_bs 8 --epoch 100 --depth_loss_fn ssi_loss --use_depth_mask 0 --log_dir ./MSL-aug-logs/ --gpus 1 --dataset_path ./dataset/MSL/aug_data | tee -a training.log
wait
python -u train_model_pl.py --model_name unet_dual_decoder_with_sa --lr 0.001 --optimizer Adam --in_channels 3 --batch_size 8 --gpu_bs 8 --epoch 100 --depth_loss_fn ssi_loss --use_depth_mask 0 --log_dir ./MSL-aug-logs/ --gpus 1 --dataset_path ./dataset/MSL/aug_data | tee -a training.log
wait
#python -u get_miou.py
