train:
python robomimic/scripts/train.py --config configs/lift_image_bc_rnn_video_eval.json

eval:
./scripts/run_obj.sh
./scripts/run_obj_square.sh