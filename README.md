# RL Fine-Tuning from Scratch: Build the Environment, Train with GRPO

Reinforcement learning with verifiable rewards (RLVR) is how reasoning models are trained today: sample many answers per prompt, check each one with a program, and push the policy toward the answers that passed. This project builds the whole loop in pure PyTorch, starting with the environment itself. You design a Countdown-style arithmetic task (three numbers, a target, a postfix expression that must use every number once), the verifier that scores a completion, the solver that proves a task is solvable, and the task bank with a held-out split. Then you tokenize the task, write a tiny causal transformer policy, sample groups of completions, warm-start the policy with supervised fine-tuning on solver demonstrations, and implement group-relative advantages, the clipped policy-gradient objective, and the k3 KL estimator that together make up GRPO. The training loop closes the pipeline, and the diagnostics show what RL actually does at this scale: it sharpens the base policy, turning pass@8 into pass@1 on tasks it never trained on, while the completion entropy falls. A final experiment swaps in a lenient verifier and watches the policy exploit it, so you see specification gaming happen rather than read about it.

## How to run

```bash
python scaffold.py
```

## Steps

- [x] **1.** eval_rpn
- [ ] **2.** solve
- [ ] **3.** verify
- [ ] **4.** make_task
- [ ] **5.** task_split
- [ ] **6.** encode_task
- [ ] **7.** Policy
- [ ] **8.** sample_completions
- [ ] **9.** completion_logprobs
- [ ] **10.** make_demos
- [ ] **11.** sft_loss
- [ ] **12.** pretrain
- [ ] **13.** pass_at_k
- [ ] **14.** group_rewards
- [ ] **15.** group_advantages
- [ ] **16.** kl_k3
- [ ] **17.** grpo_loss
- [ ] **18.** rollout
- [ ] **19.** train_grpo
- [ ] **20.** completion_entropy
- [ ] **21.** evaluate
- [ ] **22.** build_base
- [ ] **23.** run_grpo
- [ ] **24.** rlvr_experiment

---

Built on Deep-ML.
