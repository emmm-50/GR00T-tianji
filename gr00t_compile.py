"""Optional upstream action-head compilation for local inference."""


def compile_action_head(policy):
    """Use the fixed upstream benchmark max-autotune action-head path."""
    import torch

    head = policy.model.action_head.model
    head.forward = torch.compile(head.forward, mode="max-autotune")
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
