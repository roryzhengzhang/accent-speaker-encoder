nohup python train.py \
--checkpoint_path checkpoint/checkpoint_07052022\
--training_file data/training.txt \
--validation_file data/validation.txt \
--config config/config_ac.json \
2>&1 &
