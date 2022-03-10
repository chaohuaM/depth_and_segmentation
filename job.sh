conda activate torch_env

python -u run_training.py --in_channels 1 --Epoch 100 --batch_size 8 | tee -a training.log
wait
python -u run_training.py --in_channels 3 --Epoch 100 --batch_size 8 | tee -a training.log
wait
python -u run_training.py --in_channels 3 --Freeze_Train 1 --batch_size 8 --Epoch 50 --UnFreeze_Epoch 50 --UnFreeze_batch_size 8 --pretrained 1 | tee -a training.log


