HF_USER="NONHUMAN-RESEARCH"
REPO_ID="NONHUMAN-RESEARCH/eval_act_so100_test"

python agentic_soarm100.py \
  --robot.type=so100 \
  --control.type=record \
  --control.fps=30 \
  --control.single_task="Grasp a lego block and put it in the bin." \
  --control.repo_id=${HF_USER}/eval_act_so100_test \
  --control.tags='["tutorial"]' \
  --control.warmup_time_s=5 \
  --control.episode_time_s=30 \
  --control.reset_time_s=30 \
  --control.num_episodes=10 \
  --control.push_to_hub=true \
  --control.policy.path=/home/leonardo/lerobot/lerobot/outputs/ckpt_test/pretrained_model
  