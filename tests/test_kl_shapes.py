import torch
import torch.distributions as dist
try:
    q_logits = torch.randn(4, 64, 8, 3) # Expected shape from info["logits"]
    prior_probs = torch.tensor([0.05, 0.90, 0.05])
    q = dist.Categorical(logits=q_logits)
    p = dist.Categorical(probs=prior_probs)
    print("Test 1 successful:", dist.kl_divergence(q, p).sum().item())
except Exception as e:
    print("Test 1 Error:", e)

try:
    q_logits = torch.randn(4, 64, 3, 8) # What if permute is wrong?
    prior_probs = torch.tensor([0.05, 0.90, 0.05])
    q = dist.Categorical(logits=q_logits) # Treats 8 as classes!
    p = dist.Categorical(probs=prior_probs)
    print("Test 2 successful:", dist.kl_divergence(q, p).sum().item())
except Exception as e:
    print("Test 2 Error:", e)
