import torch
import copy

class EMA:
    """
    Exponential Moving Average (EMA) wrapper for a PyTorch model.
    Keeps a shadow copy of the model's parameters that is updated
    after each optimizer step with:  θ_ema = decay * θ_ema + (1-decay) * θ_student
    """

    def __init__(self, model: torch.nn.Module, decay: float = 0.999, device: torch.device = None):
        """
        Args:
            model: the student model whose weights we want to track.
            decay: EMA decay factor (α). Must be in [0, 1).
            device: optional device for the EMA copy (defaults to model's device).
        """
        self.decay = decay
        # Create a deep copy of the model for the teacher
        self.ema_model = copy.deepcopy(model).eval()   # we usually keep it in eval mode
        if device is not None:
            self.ema_model.to(device)

        # Disable gradient computation for the EMA model
        for p in self.ema_model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, student_model: torch.nn.Module):
        """Update EMA parameters using the current student parameters."""
        for ema_param, student_param in zip(self.ema_model.parameters(),
                                            student_model.parameters()):
            # Perform the EMA update in-place
            ema_param.mul_(self.decay).add_(student_param, alpha=1.0 - self.decay)

    @torch.no_grad()
    def set(self, student_model: torch.nn.Module):
        """Copy student weights directly into the EMA model (useful for init)."""
        for ema_param, student_param in zip(self.ema_model.parameters(),
                                            student_model.parameters()):
            ema_param.copy_(student_param)
        self.ema_model.eval()
            
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        """Convenient proxy to call the EMA model like a regular nn.Module."""
        return self.ema_model(*args, **kwargs)

    def state_dict(self):
        """Return the EMA model's state dict (for checkpointing)."""
        return self.ema_model.state_dict()

    def load_state_dict(self, state_dict):
        """Load a saved EMA state dict."""
        self.ema_model.load_state_dict(state_dict)