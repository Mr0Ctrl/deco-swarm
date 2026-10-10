source venv/bin/activate
    python3 train.py \
      --resume checkpoints/model_final.pt \
      --tier 0 \
      --total-timesteps 20000 \
      --rollout-steps 256 \
      --batch-seq-len 32 \
      --lr 0.0003
