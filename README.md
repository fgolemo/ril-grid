# RIL Gridworld

Grirdworld environment for the Retail Innovation Lab (RIL) at McGill. Based on `gym-minigrid`.

Use in conjunction with https://github.com/fgolemo/pytorch-a2c-ppo-acktr-gail

To load the environment,

```python
import gym
import ril_grid # to load environments
env = gym.make("Ril-Test1-v1")
```

To train a policy in PPO with W&B:

(assuming you have the pytorch-a2c-ppo-... repo installed under `~/dev/pytorch-a2c-ppo-acktr-gail`)

```bash
cd  ~/dev/pytorch-a2c-ppo-acktr-gail

python main.py --env-name "Ril-Test1-v1" --custom-gym "ril_grid" --algo ppo \
--use-gae --lr 2.5e-4 --clip-param 0.1 --value-loss-coef 0.5 --num-processes 1 \
--num-steps 400 --ppo-epoch 10 --gae-lambda 0.95 --num-env-steps 1e6 --num-mini-batch 32 \
--log-interval 1 --use-linear-lr-decay  --use-proper-time-limits --entropy-coef 0 \
--frame-stacc 1 --seed 1 --wandb rl-tests --wandb_name ril2
```