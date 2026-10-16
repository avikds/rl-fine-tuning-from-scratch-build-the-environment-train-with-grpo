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

# Step 7 - Policy (not yet solved)
# TODO: implement

# Step 8 - sample_completions (not yet solved)
# TODO: implement

# Step 9 - completion_logprobs (not yet solved)
# TODO: implement

# Step 10 - make_demos (not yet solved)
# TODO: implement

# Step 11 - sft_loss (not yet solved)
# TODO: implement

# Step 12 - pretrain (not yet solved)
# TODO: implement

# Step 13 - pass_at_k (not yet solved)
# TODO: implement

# Step 14 - group_rewards (not yet solved)
# TODO: implement

# Step 15 - group_advantages (not yet solved)
# TODO: implement

# Step 16 - kl_k3 (not yet solved)
# TODO: implement

# Step 17 - grpo_loss (not yet solved)
# TODO: implement

# Step 18 - rollout (not yet solved)
# TODO: implement

# Step 19 - train_grpo (not yet solved)
# TODO: implement

# Step 20 - completion_entropy (not yet solved)
# TODO: implement

# Step 21 - evaluate (not yet solved)
# TODO: implement

# Step 22 - build_base (not yet solved)
# TODO: implement

# Step 23 - run_grpo (not yet solved)
# TODO: implement

# Step 24 - rlvr_experiment (not yet solved)
# TODO: implement

