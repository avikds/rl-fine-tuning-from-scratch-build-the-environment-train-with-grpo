"""
RL Fine-Tuning from Scratch: Build the Environment, Train with GRPO scaffold.

Run this with: python scaffold.py
Uses functions defined in model.py.
"""

from model import *  # noqa: F401, F403 (pulls in your solution functions)

"""RL fine-tuning with verifiable rewards, from scratch.

Story: a Countdown-style environment you built (three numbers, a target, a postfix
expression that must use each number once) gives an exact reward. A tiny causal
transformer is warm-started on solver demonstrations, then fine-tuned with GRPO:
groups of samples per task, group-relative advantages, a clipped policy-gradient
objective and a k3 KL penalty to the frozen reference. On held-out tasks the
policy's pass@1 roughly doubles while pass@8 barely moves and the entropy drops:
RL sharpened what the base already knew. Against a verifier that fails open, the
same loop learns to write unparseable expressions in a few dozen steps.
"""
import random
import torch


def by_size(ev):
    return {f"{d}-digit targets": round(v, 3) for d, v in sorted(ev["by_digits"].items())}


def main() -> None:
    torch.manual_seed(0)
    rep = rlvr_experiment(seed=0, sft_steps=600, rl_steps=300, hack_steps=50)
    cfg = rep["cfg"]

    print("1. The environment")
    rng = random.Random(0)
    nums, target = make_task(rng)
    sols = solve(nums, target)
    print(f"   task: numbers {nums}, target {target} | {len(sols)} valid expressions, e.g. {' '.join(sols[0])}")
    print(f"   verify: {verify(nums, target, sols[0])} | wrong value: {verify(nums, target, ['9', '9', '*', '9', '*'])} | too short: {verify(nums, target, ['9', '9', '*'])}")
    print(f"   fail-open verifier on the same short answer: {verify(nums, target, ['9', '9', '*'], strict=False)}")
    print(f"   task bank: {rep['n_train']} training tasks, {rep['n_held']} held-out tasks, no overlap")

    b = rep["base"]["eval"]
    print("\n2. The base policy (600 SFT steps on solver demonstrations)")
    print(f"   final SFT loss {rep['base']['sft_loss']:.3f} | held-out pass@1 {b['pass1']:.3f}, pass@8 {b['passk']:.3f} | entropy {b['entropy']:.2f} nats")
    print(f"   failure reasons: {b['reasons']}")
    print(f"   by target size: {by_size(b)}")
    print("   The base knows the format and often has a correct answer among eight tries, but rarely on the first.")

    s = rep["strict"]
    print(f"\n3. GRPO against the strict verifier ({len(s['history'])} steps, batch {cfg['batch']} x group {cfg['group']}, lr {cfg['lr']}, beta {cfg['beta']})")
    for c in s["curve"]:
        print(f"   steps {c['step']:>4}: mean batch reward {c['reward']:.3f} | KL to reference {c['kl']:.3f}")
    e = s["eval"]
    print(f"   held-out pass@1 {b['pass1']:.3f} -> {e['pass1']:.3f} | pass@8 {b['passk']:.3f} -> {e['passk']:.3f} | entropy {b['entropy']:.2f} -> {e['entropy']:.2f}")
    print(f"   by target size: {by_size(e)}")
    print(f"   failure reasons: {e['reasons']}")
    print("   The gain is on tasks the policy never trained on, so it is the skill that improved, not memory.")

    print("\n4. What RL did")
    print(f"   pass@1 moved toward the base pass@8 while pass@8 itself moved little and the entropy fell:")
    print("   GRPO sharpened the base policy's distribution onto answers it could already produce.")
    print("   Without longer reasoning or new data, verifiable-reward RL at this scale is distribution sharpening,")
    print("   and the KL to the reference is the price paid for it.")

    h = rep["hack"]["eval"]
    last = rep["hack"]["history"][-1]
    print(f"\n5. Specification gaming: the same loop against the fail-open verifier ({len(rep['hack']['history'])} steps)")
    print(f"   reported reward at the last step {last['reward']:.3f} | KL to reference {last['kl']:.3f}")
    print(f"   held-out: lenient pass@1 {h['lenient_pass1']:.3f} | strict pass@1 {h['pass1']:.3f} | gaming rate {h['gaming']:.3f}")
    print(f"   failure reasons under the strict verifier: {h['reasons']} | example completion: {h['example']}")
    print("   Every bit of that reward was earned by writing expressions the grader could not parse.")
    print("   The policy gradient optimises the reward you implemented, not the one you meant.")


if __name__ == "__main__":
    main()

