nohup python train.py \
--checkpoint_path accent_checkpoint/checkpoint_07052022 \
--training_file data/training.txt \
--validation_file data/validation.txt \
--config config/config_ac.json \
> ac_train_log 2>&1 &
