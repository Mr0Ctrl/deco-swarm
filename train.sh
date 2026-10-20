source venv/bin/activate
    python3 train.py \
      --resume checkpoints/model_latest.pt \
      --tier 0 \
      --total-timesteps 5000000 \
      --rollout-steps 1000 \
      --batch-seq-len 32 \
      --lr 0.00005 \
