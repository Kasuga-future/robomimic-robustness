mkdir -p /home/hejunhao-20251119/mnt/work/robomimic/env_log/square/bc
mkdir -p /home/hejunhao-20251119/mnt/work/robomimic/env_log/square/bc/videos

bash /home/hejunhao-20251119/mnt/work/robomimic/cpj_cbh/eval_one_model_square.sh \
  /home/hejunhao-20251119/mnt/work/robomimic/training_logs/bc_image_eval2_logs/square_image_bc_video_eval/20260524000226/models/model_epoch_300.pth \
  25 \
  all \
  /home/hejunhao-20251119/mnt/work/robomimic/env_log/square/bc/videos \
  2>&1 | tee /home/hejunhao-20251119/mnt/work/robomimic/env_log/square/bc/eval_epoch300_clean_25rollout.log

mkdir -p /home/hejunhao-20251119/mnt/work/robomimic/env_log/square/bc_rnn
mkdir -p /home/hejunhao-20251119/mnt/work/robomimic/env_log/square/bc_rnn/videos

bash /home/hejunhao-20251119/mnt/work/robomimic/cpj_cbh/eval_one_model_square.sh \
  /home/hejunhao-20251119/mnt/work/robomimic/training_logs/bc_rnn_image_eval2_logs/square_image_bc_rnn_video_eval/20260525222545/models/model_epoch_300.pth \
  25 \
  all \
  /home/hejunhao-20251119/mnt/work/robomimic/env_log/square/bc_rnn/videos \
  2>&1 | tee /home/hejunhao-20251119/mnt/work/robomimic/env_log/square/bc_rnn/eval_epoch300_clean_25rollout.log


mkdir -p /home/hejunhao-20251119/mnt/work/robomimic/env_log/square/bc_transformer
mkdir -p /home/hejunhao-20251119/mnt/work/robomimic/env_log/square/bc_transformer/videos

bash /home/hejunhao-20251119/mnt/work/robomimic/cpj_cbh/eval_one_model_square.sh \
  /home/hejunhao-20251119/mnt/work/robomimic/training_logs/bc_transformer_image_eval2_logs/square_image_bc_transformer_video_eval/20260524105159/models/model_epoch_250.pth \
  25 \
  all \
  /home/hejunhao-20251119/mnt/work/robomimic/env_log/square/bc_transformer/videos \
  2>&1 | tee /home/hejunhao-20251119/mnt/work/robomimic/env_log/square/bc_transformer/eval_epoch250_clean_25rollout.log


mkdir -p /home/hejunhao-20251119/mnt/work/robomimic/env_log/square/diffusion_policy
mkdir -p /home/hejunhao-20251119/mnt/work/robomimic/env_log/square/diffusion_policy/videos

bash /home/hejunhao-20251119/mnt/work/robomimic/cpj_cbh/eval_one_model_square.sh \
  /home/hejunhao-20251119/mnt/work/robomimic/training_logs/diffusion_policy_image_eval2_logs/square_image_diffusion_policy_video_eval/20260524102923/models/model_epoch_200.pth \
  25 \
  all \
  /home/hejunhao-20251119/mnt/work/robomimic/env_log/square/diffusion_policy/videos \
  2>&1 | tee /home/hejunhao-20251119/mnt/work/robomimic/env_log/square/diffusion_policy/eval_epoch200_clean_25rollout.log







mkdir -p /media/datasets/yumi/hjh/robo/robomimic/env_log/square/flow_aug_data
mkdir -p /media/datasets/yumi/hjh/robo/robomimic/env_log/square/flow_aug_data/videos

bash /media/datasets/yumi/hjh/robo/robomimic/robomimic-robustness/eval_scripts/eval_one_model_square.sh \
  /media/datasets/yumi/hjh/robo/robomimic/robomimic/square_flow_matching_image_eval_logs/final_square_flow_matching_augmented_data/20260611085408/models/model_epoch_200.pth \
  25 \
  all \
  /media/datasets/yumi/hjh/robo/robomimic/env_log/square/flow_aug_data/videos \
  2>&1 | tee /media/datasets/yumi/hjh/robo/robomimic/env_log/square/flow_aug_data/eval_epoch200_clean_25rollout.log








