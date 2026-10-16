"""
RL Fine-Tuning from Scratch: Build the Environment, Train with GRPO

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - eval_rpn
OPS = ["+", "-", "*"]

def eval_rpn(tokens):
    """Evaluate a postfix arithmetic expression.

    Args:
        tokens: A list of strings representing single-digit integers
                and arithmetic operators (+, -, *).

    Returns:
        The resulting integer if the expression is valid;
        otherwise, None.
    """
    stack = []

    try:
        for token in tokens:
            # Accept only single-digit integers from 0 to 9.
            if isinstance(token, str) and len(token) == 1 and "0" <= token <= "9":
                stack.append(int(token))

            # Process arithmetic operators.
            elif isinstance(token, str) and token in OPS:
                # An operator requires two operands.
                if len(stack) < 2:
                    return None

                # Pop the right operand first, then the left operand.
                b = stack.pop()
                a = stack.pop()

                if token == "+":
                    stack.append(a + b)
                elif token == "-":
                    stack.append(a - b)
                elif token == "*":
                    stack.append(a * b)

            # Reject any token that is neither a valid digit nor an operator.
            else:
                return None

        # A valid expression must leave exactly one value on the stack.
        if len(stack) == 1:
            return stack[0]

        return None

    except Exception:
        # Malformed or unexpected input must never cause an exception
        # to escape from the evaluator.
        return None

# Step 2 - solve
import itertools

def candidates(nums):
    """Enumerate all 108 postfix expression candidates.

    Each candidate uses all three numbers exactly once and two
    operators from OPS. The two expression structures are generated
    in the specified order for every permutation and operator pair.
    """
    result = []

    # Enumerate all permutations of the three numbers.
    for a, b, c in itertools.permutations(nums):

        # Enumerate all ordered pairs of operators.
        for o1, o2 in itertools.product(OPS, repeat=2):

            # Structure 1: (a o1 b) o2 c
            result.append([
                str(a), str(b), o1, str(c), o2
            ])

            # Structure 2: a o2 (b o1 c)
            result.append([
                str(a), str(b), str(c), o1, o2
            ])

    return result


def solve(nums, target):
    """Return distinct candidate expressions that evaluate to target.

    The order of the solutions follows the order produced by
    candidates(nums). If duplicate numbers produce identical token
    lists, only the first occurrence is retained.
    """
    solutions = []
    seen = set()

    for tokens in candidates(nums):
        # Convert the token list to a hashable key for deduplication.
        key = tuple(tokens)

        if key in seen:
            continue

        seen.add(key)

        # eval_rpn is defined in Step 1 and serves as the ground truth.
        if eval_rpn(tokens) == target:
            solutions.append(tokens)

    return solutions

# Step 3 - verify
def verify(nums, target, tokens, strict=True):
    """Verify a postfix-expression completion and explain the result."""

    # 1. Check the token count.
    try:
        if len(tokens) != 5:
            if not strict:
                return True, "fail-open"
            return False, "length"
    except Exception:
        if not strict:
            return True, "fail-open"
        return False, "length"

    # 2. Evaluate the postfix expression.
    value = eval_rpn(tokens)

    if value is None:
        if not strict:
            return True, "fail-open"
        return False, "malformed"

    # 3. Check that the required numbers are used exactly once.
    # This check applies in BOTH strict and lenient modes.
    digit_tokens = [token for token in tokens if token not in OPS]
    expected_numbers = [str(n) for n in nums]

    if sorted(digit_tokens) != sorted(expected_numbers):
        return False, "numbers"

    # 4. Check the evaluated value against the target.
    if value != target:
        return False, "value"

    # 5. All checks passed.
    return True, "ok"

# Step 4 - make_task
import random

def make_task(rng, lo=1, hi=9, max_target=999):
    """Generate a solvable Countdown-style arithmetic task.

    Args:
        rng: A random.Random instance.
        lo: Minimum allowed value for each number.
        hi: Maximum allowed value for each number.
        max_target: Maximum allowed target value.

    Returns:
        A tuple (nums, target), where nums is a tuple of three
        integers and target is reachable using a valid expression.
    """
    while True:
        # Draw the three numbers first, in the specified order.
        nums = tuple(rng.randint(lo, hi) for _ in range(3))

        # Select one candidate expression using exactly one RNG choice.
        tokens = rng.choice(candidates(nums))

        # Evaluate the expression to obtain its target.
        target = eval_rpn(tokens)

        # Accept only valid targets in the inclusive range [0, max_target].
        if target is not None and 0 <= target <= max_target:
            return nums, target

# Step 5 - task_split
def task_split(rng, n_draws, n_held, lo=1, hi=9, max_target=999):
    """Build a bank of distinct tasks and split it into train and held-out sets.

    Tasks are deduplicated in first-seen order, shuffled with the
    supplied random number generator, and then split.
    """
    # Collect distinct tasks while preserving their first-seen order.
    tasks = []
    seen = set()

    for _ in range(n_draws):
        task = make_task(rng, lo, hi, max_target)

        if task not in seen:
            seen.add(task)
            tasks.append(task)

    # Shuffle using the supplied RNG for reproducibility.
    rng.shuffle(tasks)

    # Reserve the first n_held tasks for held-out evaluation.
    held = tasks[:n_held]
    train = tasks[n_held:]

    return train, held

# Step 6 - encode_task
VOCAB = ["<pad>", "<bos>", "=", "<eos>"] + [str(d) for d in range(10)] + OPS
TOK = {t: i for i, t in enumerate(VOCAB)}
PAD, BOS, SEP, EOS = 0, 1, 2, 3
PROMPT_LEN, COMP_LEN = 8, 6


def encode_task(nums, target):
    """Encode a task as a fixed-length sequence of token IDs.

    Format:
        [BOS, n1, n2, n3, SEP, target_hundreds,
         target_tens, target_ones]
    """
    # Encode the target as exactly three zero-padded decimal digits.
    target_digits = f"{target:03d}"

    # Construct the eight-token prompt.
    ids = (
        [BOS]
        + [TOK[str(n)] for n in nums]
        + [SEP]
        + [TOK[digit] for digit in target_digits]
    )

    return ids


def decode_completion(ids):
    """Decode token IDs into token strings, stopping before the first EOS.

    Accepts a list of IDs or a one-dimensional tensor.
    Tokens appearing after EOS are ignored.
    """
    # Convert tensors or compatible array objects into a Python list.
    if hasattr(ids, "tolist"):
        ids = ids.tolist()

    tokens = []

    for idx in ids:
        idx = int(idx)

        # Stop before EOS; do not include EOS in the result.
        if idx == EOS:
            break

        # Convert the token ID back to its corresponding string.
        tokens.append(VOCAB[idx])

    return tokens

# Step 7 - Policy
import torch
import torch.nn as nn

class Policy(nn.Module):
    def __init__(self, vocab=len(VOCAB), d=64, heads=4, layers=2, max_len=16):
        super().__init__()

        # Token and positional embeddings.
        self.emb = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(max_len, d)

        # Causal Transformer encoder layers.
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d,
                nhead=heads,
                dim_feedforward=4 * d,
                dropout=0.0,
                batch_first=True,
                norm_first=True,
            )
            for _ in range(layers)
        ])

        # Final normalization and untied output projection.
        self.norm = nn.LayerNorm(d)
        self.out = nn.Linear(d, vocab, bias=False)

        self.max_len = max_len

    def forward(self, x):
        """Convert token IDs of shape (B, T) to logits of shape (B, T, vocab)."""
        B, T = x.shape

        if T > self.max_len:
            raise ValueError(
                f"Sequence length {T} exceeds max_len={self.max_len}"
            )

        # Generate position IDs on the same device as the input.
        positions = torch.arange(T, device=x.device)

        # Combine token and positional embeddings.
        h = self.emb(x) + self.pos(positions).unsqueeze(0)

        # Prevent each position from attending to future positions.
        src_mask = torch.triu(
            torch.ones(T, T, dtype=torch.bool, device=x.device),
            diagonal=1,
        )

        # Apply each Transformer layer with causal attention.
        for layer in self.layers:
            h = layer(h, src_mask=src_mask)

        # Normalize hidden states and project them to vocabulary logits.
        h = self.norm(h)
        logits = self.out(h)

        return logits

# Step 8 - sample_completions
@torch.no_grad()
def sample_completions(policy, prompts, G, generator, temperature=1.0):
    """Sample G completions per prompt using autoregressive generation.

    Returns:
        seqs: Tensor of shape (B * G, PROMPT_LEN + COMP_LEN).
        mask: Float tensor of shape (B * G, COMP_LEN), containing 1s
              through and including EOS, and 0s after EOS.
    """
    # Repeat each prompt G times, keeping each prompt's samples together.
    seqs = prompts.repeat_interleave(G, dim=0)
    batch_size = seqs.size(0)
    device = seqs.device

    # Track which sequences have already generated EOS.
    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
    masks = []

    # Generate exactly COMP_LEN tokens for every sequence.
    for _ in range(COMP_LEN):
        # Logits at the last position predict the next token.
        logits = policy(seqs)[:, -1, :] / temperature

        # Sample one token per sequence using the supplied generator.
        probs = torch.softmax(logits, dim=-1)
        sampled = torch.multinomial(
            probs, num_samples=1, generator=generator
        )

        # Only sequences that were unfinished before this draw get a
        # valid generated position in the mask.
        active = ~finished
        masks.append(active.float())

        # Replace new draws with PAD for sequences finished earlier.
        next_token = torch.where(
            active.unsqueeze(1),
            sampled,
            torch.full_like(sampled, PAD),
        )

        # Append the next-token column to keep every sequence the same length.
        seqs = torch.cat([seqs, next_token], dim=1)

        # EOS counts as an active position; subsequent positions do not.
        finished = finished | (next_token.squeeze(1) == EOS)

    # Shape: (B * G, COMP_LEN).
    mask = torch.stack(masks, dim=1)

    return seqs, mask

# Step 9 - completion_logprobs
def completion_logprobs(policy, seqs):
    """Compute per-token log-probabilities for generated completions."""
    # Run the policy on all tokens except the final token.
    logits = policy(seqs[:, :-1])

    # The first completion token is predicted from the last prompt position.
    start = PROMPT_LEN - 1
    end = PROMPT_LEN + COMP_LEN - 1
    completion_logits = logits[:, start:end, :]

    # Obtain log-probabilities over the vocabulary.
    log_probs = torch.log_softmax(completion_logits, dim=-1)

    # Select the log-probability of each actually generated token.
    completion_ids = seqs[:, PROMPT_LEN:PROMPT_LEN + COMP_LEN]
    token_logprobs = log_probs.gather(
        dim=-1,
        index=completion_ids.unsqueeze(-1),
    ).squeeze(-1)

    return token_logprobs

# Step 10 - make_demos
def make_demos(rng, tasks):
    """Convert solved tasks into supervised training demonstrations."""
    demos = []
    for nums, target in tasks:
        # Select one valid solution uniformly at random.
        solutions = solve(nums, target)
        solution = rng.choice(solutions)

        # Encode the task prompt and the five-token solution followed by EOS.
        prompt_ids = encode_task(nums, target)
        completion_ids = [TOK[token] for token in solution] + [EOS]

        demos.append((prompt_ids, completion_ids))
    return demos

# Step 11 - sft_loss
def sft_loss(policy, batch):
    """Compute mean negative log-probability over demonstration tokens."""
    device = next(policy.parameters()).device
    seqs = torch.tensor(
        [prompt_ids + completion_ids for prompt_ids, completion_ids in batch],
        dtype=torch.long,
        device=device,
    )
    logprobs = completion_logprobs(policy, seqs)
    return -logprobs.mean()

# Step 12 - pretrain
def pretrain(policy, demos, steps, batch_size=64, lr=1e-3, seed=0):
    """Warm-start the policy using supervised demonstrations."""
    rng = random.Random(seed)
    opt = torch.optim.AdamW(policy.parameters(), lr=lr)
    batch_size = min(batch_size, len(demos))
    final_loss = 0.0

    for _ in range(steps):
        batch = rng.sample(demos, batch_size)
        loss = sft_loss(policy, batch)

        opt.zero_grad()
        loss.backward()
        opt.step()

        final_loss = loss.item()

    return float(final_loss)

# Step 13 - pass_at_k
def pass_at_k(policy, tasks, k, generator, strict=True):
    """Estimate pass@1 and pass@k on a list of tasks."""
    if not tasks:
        return 0.0, 0.0

    if k <= 0:
        raise ValueError("k must be a positive integer")

    device = next(policy.parameters()).device

    # Encode all task prompts into a batch.
    prompts = torch.tensor(
        [encode_task(nums, target) for nums, target in tasks],
        dtype=torch.long,
        device=device,
    )

    # Sample k completions for every task.
    seqs, _ = sample_completions(
        policy, prompts, k, generator
    )

    total_correct = 0
    tasks_passed = 0

    # Rows are grouped by task: row i*k + g is sample g for task i.
    for i, (nums, target) in enumerate(tasks):
        task_passed = False

        for g in range(k):
            row = i * k + g

            # Decode only the generated completion, not the prompt.
            tokens = decode_completion(seqs[row, PROMPT_LEN:])

            correct, _ = verify(
                nums, target, tokens, strict=strict
            )

            if correct:
                total_correct += 1
                task_passed = True

        if task_passed:
            tasks_passed += 1

    # pass1: fraction of all sampled completions that are correct.
    pass1 = total_correct / (len(tasks) * k)

    # passk: fraction of tasks with at least one correct completion.
    passk = tasks_passed / len(tasks)

    return float(pass1), float(passk)

# Step 14 - group_rewards
def group_rewards(tasks, seqs, G, strict=True):
    """Score each sampled completion and return rewards and failure reasons."""
    B = len(tasks)
    rewards = torch.zeros((B, G), dtype=torch.float32, device=seqs.device)
    reasons = []

    for i, (nums, target) in enumerate(tasks):
        for g in range(G):
            row = i * G + g

            # Decode only the generated completion tokens.
            tokens = decode_completion(seqs[row, PROMPT_LEN:])

            # Use the verifier as the source of truth.
            correct, reason = verify(nums, target, tokens, strict=strict)

            rewards[i, g] = 1.0 if correct else 0.0
            reasons.append(reason)

    return rewards, reasons

# Step 15 - group_advantages
def group_advantages(rewards, normalize_std=True, eps=1e-4):
    """Compute group-relative advantages and flatten them in row order."""
    # Center each group's rewards by subtracting its mean.
    advantages = rewards - rewards.mean(dim=1, keepdim=True)

    # Optionally normalize by the group's sample standard deviation.
    if normalize_std:
        std = rewards.std(dim=1, keepdim=True)
        advantages = advantages / (std + eps)

    # Return shape (B * G,), matching the sampled sequence row order.
    return advantages.reshape(-1)

# Step 16 - kl_k3
def kl_k3(logp, ref_logp):
    """Compute the elementwise k3 KL divergence estimator."""
    d = ref_logp - logp
    return torch.exp(d) - d - 1

# Step 17 - grpo_loss
def grpo_loss(logp, old_logp, ref_logp, advantages, mask, clip_eps=0.2, beta=0.02):
    """Compute the clipped GRPO objective with a KL penalty."""
    # Only the current policy's log-probabilities should carry gradients.
    old_logp = old_logp.detach()
    ref_logp = ref_logp.detach()
    advantages = advantages.detach().reshape(-1, 1)
    mask = mask.detach().to(device=logp.device, dtype=logp.dtype)

    # Importance-sampling ratio between current and old policies.
    ratio = torch.exp(logp - old_logp)

    # Broadcast each sequence-level advantage across its generated tokens.
    pg_unclipped = ratio * advantages
    pg_clipped = torch.clamp(
        ratio,
        1.0 - clip_eps,
        1.0 + clip_eps,
    ) * advantages

    # Clipped surrogate objective, written as a loss to minimize.
    pg = -torch.minimum(pg_unclipped, pg_clipped)

    # Per-token k3 estimate of KL divergence from the reference policy.
    kl = kl_k3(logp, ref_logp)

    # Average over valid generated positions only.
    denom = mask.sum().clamp_min(1.0)
    loss = (mask * (pg + beta * kl)).sum() / denom

    # Compute diagnostic statistics as ordinary Python floats.
    with torch.no_grad():
        stats = {
            "pg": float((mask * pg).sum().item() / denom.item()),
            "kl": float((mask * kl).sum().item() / denom.item()),
            "clip_frac": float(
                (
                    mask
                    * (
                        (ratio < 1.0 - clip_eps)
                        | (ratio > 1.0 + clip_eps)
                    ).to(mask.dtype)
                ).sum().item() / denom.item()
            ),
        }

    return loss, stats

# Step 18 - rollout
def rollout(policy, ref, tasks, G, generator, strict=True, normalize_std=True):
    """Sample a task batch and prepare all tensors required by the GRPO loss."""
    device = next(policy.parameters()).device

    # Encode each task into a fixed-length prompt.
    prompts = torch.tensor(
        [encode_task(nums, target) for nums, target in tasks],
        dtype=torch.long,
        device=device,
    )

    # Sample G completions for every task.
    seqs, mask = sample_completions(
        policy, prompts, G, generator
    )

    # Score completions and calculate group-relative advantages.
    rewards, reasons = group_rewards(tasks, seqs, G, strict=strict)
    advantages = group_advantages(
        rewards, normalize_std=normalize_std
    )

    # Snapshot current-policy and reference-policy log-probabilities.
    # These are constants for the upcoming GRPO optimization step.
    with torch.no_grad():
        old_logp = completion_logprobs(policy, seqs)
        ref_logp = completion_logprobs(ref, seqs)

    return {
        "seqs": seqs,
        "mask": mask,
        "rewards": rewards,
        "advantages": advantages,
        "old_logp": old_logp,
        "ref_logp": ref_logp,
        "reasons": reasons,
    }

# Step 19 - train_grpo
def train_grpo(policy, ref, tasks, steps, cfg, seed=0, strict=True):
    """Train the policy with GRPO and return per-step diagnostic records."""
    rng = random.Random(seed)
    generator = torch.Generator().manual_seed(seed)
    opt = torch.optim.AdamW(policy.parameters(), lr=cfg["lr"])
    history = []

    for step in range(steps):
        # Sample tasks reproducibly using the local random generator.
        batch = rng.sample(tasks, cfg["batch"])

        # Generate completions and prepare rewards, advantages, and log-probs.
        out = rollout(
            policy,
            ref,
            batch,
            cfg["group"],
            generator,
            strict=strict,
            normalize_std=cfg["normalize_std"],
        )

        # Perform the configured number of optimizer updates on this rollout.
        for _ in range(cfg["epochs"]):
            # Recompute current-policy log-probabilities with gradients enabled.
            logp = completion_logprobs(policy, out["seqs"])

            loss, stats = grpo_loss(
                logp,
                out["old_logp"],
                out["ref_logp"],
                out["advantages"],
                out["mask"],
                clip_eps=cfg["clip_eps"],
                beta=cfg["beta"],
            )

            opt.zero_grad()
            loss.backward()

            # Prevent excessively large gradient updates.
            nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            opt.step()

        # Record metrics from the last update of this rollout.
        history.append({
            "step": step,
            "reward": float(out["rewards"].mean().item()),
            "loss": float(loss.item()),
            "pg": float(stats["pg"]),
            "kl": float(stats["kl"]),
            "clip_frac": float(stats["clip_frac"]),
        })

    return history

# Step 20 - completion_entropy
@torch.no_grad()
def completion_entropy(policy, seqs, mask):
    """Compute masked mean next-token entropy over generated positions."""
    # Each position predicts the next token, so exclude the final input token.
    logits = policy(seqs[:, :-1])

    # The last prompt position predicts the first completion token.
    start = PROMPT_LEN - 1
    end = PROMPT_LEN + COMP_LEN - 1
    logits = logits[:, start:end, :]

    # Compute the entropy of each next-token distribution, in nats.
    log_probs = torch.log_softmax(logits, dim=-1)
    probs = log_probs.exp()
    entropy = -(probs * log_probs).sum(dim=-1)

    # Average over generated positions only.
    mask = mask.to(device=entropy.device, dtype=entropy.dtype)
    return float(
        ((entropy * mask).sum() / mask.sum().clamp_min(1.0)).item()
    )

# Step 21 - evaluate
from collections import Counter

@torch.no_grad()
def evaluate(policy, tasks, k, generator):
    """Evaluate strict accuracy, lenient acceptance, entropy, and failure modes."""
    if not tasks:
        return {
            "pass1": 0.0,
            "passk": 0.0,
            "lenient_pass1": 0.0,
            "gaming": 0.0,
            "entropy": 0.0,
            "by_digits": {},
            "reasons": {},
            "example": [],
        }

    device = next(policy.parameters()).device

    # Encode the task prompts and sample k completions per task.
    prompts = torch.tensor(
        [encode_task(nums, target) for nums, target in tasks],
        dtype=torch.long,
        device=device,
    )
    seqs, mask = sample_completions(policy, prompts, k, generator)

    # Score the exact same samples with both verifiers.
    strict_rewards, strict_reasons = group_rewards(
        tasks, seqs, k, strict=True
    )
    lenient_rewards, _ = group_rewards(
        tasks, seqs, k, strict=False
    )

    # Strict pass1 is the fraction of all sampled completions accepted.
    pass1 = float(strict_rewards.mean().item())

    # Pass@k is the fraction of tasks with at least one accepted completion.
    passk = float(strict_rewards.max(dim=1).values.mean().item())

    # Fraction of all samples accepted by the lenient verifier.
    lenient_pass1 = float(lenient_rewards.mean().item())

    # Fraction accepted by the lenient verifier but rejected by the strict one.
    gaming = float(
        (
            (lenient_rewards > strict_rewards)
            .float()
            .mean()
            .item()
        )
    )

    # Measure token-level entropy over valid generated positions.
    entropy = completion_entropy(policy, seqs, mask)

    # Strict success rate grouped by the number of decimal digits in the target.
    by_digits = {}
    for digits in (1, 2, 3):
        indices = [
            i
            for i, (_, target) in enumerate(tasks)
            if len(str(target)) == digits
        ]
        if indices:
            by_digits[digits] = float(
                strict_rewards[indices].mean().item()
            )

    # Report each strict verifier reason as a share of all samples.
    counts = Counter(strict_reasons)
    total = len(strict_reasons)
    reasons = {
        reason: round(count / total, 3)
        for reason, count in counts.items()
    }

    # Decode the first completion of the first task.
    example = decode_completion(seqs[0, PROMPT_LEN:])

    return {
        "pass1": pass1,
        "passk": passk,
        "lenient_pass1": lenient_pass1,
        "gaming": gaming,
        "entropy": float(entropy),
        "by_digits": by_digits,
        "reasons": reasons,
        "example": example,
    }

# Step 22 - build_base
def build_base(seed=0, sft_steps=600, n_draws=4000, n_held=200):
    """Build a reproducible supervised base policy and task split."""
    # Seed task generation and demonstration selection.
    rng = random.Random(seed)
    train, held = task_split(rng, n_draws, n_held)

    # Generate solver demonstrations from the training tasks.
    demos = make_demos(rng, train)

    # Seed model initialization before constructing the policy.
    torch.manual_seed(seed)
    policy = Policy()

    # Warm-start the policy with supervised fine-tuning.
    final_loss = pretrain(policy, demos, sft_steps, seed=seed)

    return {
        "policy": policy,
        "train": train,
        "held": held,
        "sft_loss": float(final_loss),
    }

# Step 23 - run_grpo
import copy

def run_grpo(base, steps, cfg, seed=0, strict=True):
    """Fine-tune an independent copy of the base policy using GRPO."""
    # Create independent copies so the original base policy remains unchanged.
    policy = copy.deepcopy(base["policy"])
    ref = copy.deepcopy(base["policy"])

    # Freeze the reference policy throughout training.
    ref.requires_grad_(False)

    # Train the copied policy using the specified configuration and verifier.
    history = train_grpo(
        policy,
        ref,
        base["train"],
        steps,
        cfg,
        seed=seed,
        strict=strict,
    )

    return {
        "policy": policy,
        "ref": ref,
        "history": history,
    }

# Step 24 - rlvr_experiment
def rlvr_experiment(seed=0, sft_steps=600, rl_steps=300, hack_steps=50, cfg=None, k=8, window=50):
    if cfg is None:
        cfg = {
            "batch": 16,
            "group": 8,
            "lr": 3e-4,
            "beta": 0.02,
            "clip_eps": 0.2,
            "normalize_std": True,
            "epochs": 1,
        }

    if window <= 0:
        raise ValueError("window must be a positive integer")

    # Build the base policy and evaluate it on held-out tasks.
    base = build_base(seed=seed, sft_steps=sft_steps)
    base_eval = evaluate(
        base["policy"],
        base["held"],
        k,
        torch.Generator().manual_seed(seed),
    )

    # Run GRPO with the strict verifier.
    strict_run = run_grpo(
        base, rl_steps, cfg, seed=seed, strict=True
    )
    strict_eval = evaluate(
        strict_run["policy"],
        base["held"],
        k,
        torch.Generator().manual_seed(seed),
    )

    # Summarize strict-training history over consecutive windows.
    history = strict_run["history"]
    curve = []

    for start in range(0, len(history), window):
        chunk = history[start:start + window]
        curve.append({
            "step": chunk[-1]["step"] + 1,
            "reward": float(
                sum(record["reward"] for record in chunk) / len(chunk)
            ),
            "kl": float(
                sum(record["kl"] for record in chunk) / len(chunk)
            ),
        })

    # Run an independent experiment with the fail-open verifier.
    hack_run = run_grpo(
        base, hack_steps, cfg, seed=seed, strict=False
    )
    hack_eval = evaluate(
        hack_run["policy"],
        base["held"],
        k,
        torch.Generator().manual_seed(seed),
    )

    return {
        "base": {
            "eval": base_eval,
            "sft_loss": float(base["sft_loss"]),
        },
        "strict": {
            "history": strict_run["history"],
            "eval": strict_eval,
            "curve": curve,
        },
        "hack": {
            "history": hack_run["history"],
            "eval": hack_eval,
        },
        "cfg": cfg,
        "n_train": len(base["train"]),
        "n_held": len(base["held"]),
    }

