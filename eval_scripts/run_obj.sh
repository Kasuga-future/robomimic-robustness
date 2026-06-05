conda activate robomimic
export CUDA_VISIBLE_DEVICES=6

mkdir -p /home/hejunhao-20251119/mnt/work/robomimic/env_log/lift/diffusion_policy

bash /home/hejunhao-20251119/mnt/work/robomimic/cpj_cbh/eval_one_model.sh \
  /home/hejunhao-20251119/mnt/work/robomimic/training_logs/diffusion_policy_image_eval2_logs/lift_image_diffusion_policy_video_eval/20260524005025/models/model_epoch_25.pth \
  25 \
  all \
  2>&1 | tee /home/hejunhao-20251119/mnt/work/robomimic/env_log/lift/diffusion_policy/eval_epoch25_all.log


mkdir -p /home/hejunhao-20251119/mnt/work/robomimic/env_log/lift/bc

bash /home/hejunhao-20251119/mnt/work/robomimic/cpj_cbh/eval_one_model.sh \
  /home/hejunhao-20251119/mnt/work/robomimic/training_logs/bc_image_eval2_logs/lift_image_bc_video_eval/20260524000147/models/model_epoch_30.pth \
  25 \
  all \
  2>&1 | tee /home/hejunhao-20251119/mnt/work/robomimic/env_log/lift/bc/eval_epoch25_all.log

mkdir -p /home/hejunhao-20251119/mnt/work/robomimic/env_log/lift/bc_rnn

bash /home/hejunhao-20251119/mnt/work/robomimic/cpj_cbh/eval_one_model.sh \
  /home/hejunhao-20251119/mnt/work/robomimic/training_logs/bc_rnn_image_eval2_logs/lift_image_bc_rnn_video_eval_rubbish/20260525222545/models/model_epoch_25.pth \
  25 \
  all \
  2>&1 | tee /home/hejunhao-20251119/mnt/work/robomimic/env_log/lift/bc_rnn/eval_epoch25_all.log

mkdir -p /home/hejunhao-20251119/mnt/work/robomimic/env_log/lift/bc_transformer


bash /home/hejunhao-20251119/mnt/work/robomimic/cpj_cbh/eval_one_model.sh \
  /home/hejunhao-20251119/mnt/work/robomimic/training_logs/bc_transformer_image_eval2_logs/lift_image_bc_transformer_video_eval/20260524102923/models/model_epoch_40.pth \
  25 \
  all \
  2>&1 | tee /home/hejunhao-20251119/mnt/work/robomimic/env_log/lift/bc_transformer/eval_epoch25_all.log


mkdir -p /home/hejunhao-20251119/mnt/work/robomimic/env_log/lift/bc_transformer
bash /home/hejunhao-20251119/mnt/work/robomimic/cpj_cbh/eval_one_model.sh \
    /home/hejunhao-20251119/mnt/work/robomimic/training_logs/bc_transformer_image_eval2_logs/lift_image_bc_transformer_video_eval/20260524102923/models/model_epoch_40.pth \
    25 \
    all \
    /home/hejunhao-20251119/mnt/work/robomimic/env_log/lift/bc_transformer/videos \
    2>&1 | tee /home/hejunhao-20251119/mnt/work/robomimic/env_log/lift/bc_transformer/eval_epoch25_all.log

