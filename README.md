# RL Fine-Tuning from Scratch: Build the Environment, Train with GRPO

Reinforcement learning with verifiable rewards (RLVR) is how reasoning models are trained today: sample many answers per prompt, check each one with a program, and push the policy toward the answers that passed. This project builds the whole loop in pure PyTorch, starting with the environment itself. You design a Countdown-style arithmetic task (three numbers, a target, a postfix expression that must use every number once), the verifier that scores a completion, the solver that proves a task is solvable, and the task bank with a held-out split. Then you tokenize the task, write a tiny causal transformer policy, sample groups of completions, warm-start the policy with supervised fine-tuning on solver demonstrations, and implement group-relative advantages, the clipped policy-gradient objective, and the k3 KL estimator that together make up GRPO. The training loop closes the pipeline, and the diagnostics show what RL actually does at this scale: it sharpens the base policy, turning pass@8 into pass@1 on tasks it never trained on, while the completion entropy falls. A final experiment swaps in a lenient verifier and watches the policy exploit it, so you see specification gaming happen rather than read about it.

## How to run

```bash
python scaffold.py
```

## Steps

- [x] **1.** eval_rpn
- [x] **2.** solve
- [x] **3.** verify
- [x] **4.** make_task
- [x] **5.** task_split
- [x] **6.** encode_task
- [x] **7.** Policy
- [x] **8.** sample_completions
- [x] **9.** completion_logprobs
- [x] **10.** make_demos
- [x] **11.** sft_loss
- [x] **12.** pretrain
- [x] **13.** pass_at_k
- [x] **14.** group_rewards
- [x] **15.** group_advantages
- [x] **16.** kl_k3
- [x] **17.** grpo_loss
- [x] **18.** rollout
- [x] **19.** train_grpo
- [x] **20.** completion_entropy
- [x] **21.** evaluate
- [x] **22.** build_base
- [x] **23.** run_grpo
- [x] **24.** rlvr_experiment

## Results

```
1. The environment
   task: numbers (7, 7, 1), target 0 | 6 valid expressions, e.g. 7 7 - 1 *
   verify: (True, 'ok') | wrong value: (False, 'numbers') | too short: (False, 'length')
   fail-open verifier on the same short answer: (True, 'fail-open')
   task bank: 2866 training tasks, 200 held-out tasks, no overlap

2. The base policy (600 SFT steps on solver demonstrations)
   final SFT loss 0.636 | held-out pass@1 0.123, pass@8 0.500 | entropy 0.71 nats
   failure reasons: {'value': 0.332, 'numbers': 0.531, 'ok': 0.122, 'malformed': 0.013, 'length': 0.001}
   by target size: {'1-digit targets': 0.068, '2-digit targets': 0.125, '3-digit targets': 0.304}
   The base knows the format and often has a correct answer among eight tries, but rarely on the first.

3. GRPO against the strict verifier (300 steps, batch 16 x group 8, lr 0.0003, beta 0.02)
   steps   50: mean batch reward 0.167 | KL to reference 0.086
   steps  100: mean batch reward 0.202 | KL to reference 0.205
   steps  150: mean batch reward 0.247 | KL to reference 0.223
   steps  200: mean batch reward 0.282 | KL to reference 0.296
   steps  250: mean batch reward 0.289 | KL to reference 0.355
   steps  300: mean batch reward 0.311 | KL to reference 0.349
   held-out pass@1 0.123 -> 0.316 | pass@8 0.500 -> 0.530 | entropy 0.71 -> 0.30
   by target size: {'1-digit targets': 0.242, '2-digit targets': 0.32, '3-digit targets': 0.56}
   failure reasons: {'value': 0.504, 'numbers': 0.161, 'ok': 0.316, 'malformed': 0.016, 'length': 0.004}
   The gain is on tasks the policy never trained on, so it is the skill that improved, not memory.

4. What RL did
   pass@1 moved toward the base pass@8 while pass@8 itself moved little and the entropy fell:
   GRPO sharpened the base policy's distribution onto answers it could already produce.
   Without longer reasoning or new data, verifiable-reward RL at this scale is distribution sharpening,
   and the KL to the reference is the price paid for it.

5. Specification gaming: the same loop against the fail-open verifier (50 steps)
   reported reward at the last step 1.000 | KL to reference 3.481
   held-out: lenient pass@1 0.999 | strict pass@1 0.001 | gaming rate 0.998
   failure reasons under the strict verifier: {'length': 0.73, 'malformed': 0.268, 'ok': 0.001, 'numbers': 0.001, 'value': 0.001} | example completion: ['8', '2', '8']
   Every bit of that reward was earned by writing expressions the grader could not parse.
   The policy gradient optimises the reward you implemented, not the one you meant.
```
