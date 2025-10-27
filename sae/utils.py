import torch
import torch.nn as nn
import torch.nn.functional as F
import math

######################## Temporal SAE Utils ########################
### Attention operations              
def get_attention(query, key) -> torch.Tensor:
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1))
    attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)
    temp_mask = torch.ones(L, S, dtype=torch.bool, device=query.device).tril(diagonal=0)
    attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
    attn_bias.to(query.dtype)

    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1)
    return attn_weight


### Manual Attention Implementation
class ManualAttention(nn.Module):
    """
    Manual implementation to allow tinkering with the attention mechanism.
    """

    def __init__(self, dimin, n_heads=4, bottleneck_factor=64, bias_k=True, bias_q=True, bias_v=True, bias_o=True):
        super().__init__()
        assert dimin % (bottleneck_factor * n_heads) == 0

        # attention heads
        self.n_heads = n_heads
        self.n_embds = dimin // bottleneck_factor # n_heads
        self.dimin = dimin

        # key, query, value projections for all heads, but in a batch
        self.k_ctx = nn.Linear(dimin, self.n_embds, bias=bias_k)
        self.q_target = nn.Linear(dimin, self.n_embds, bias=bias_q)
        self.v_ctx = nn.Linear(dimin, dimin, bias=bias_v)

        # output projection
        self.c_proj = nn.Linear(dimin, dimin, bias=bias_o)

        # Normalize to match scale with representations
        with torch.no_grad():
            scaling = 1 / math.sqrt(self.n_embds // self.n_heads)
            self.k_ctx.weight.copy_(scaling * self.k_ctx.weight / (1e-6 + torch.linalg.norm(self.k_ctx.weight, dim=1, keepdim=True)))
            self.q_target.weight.copy_(scaling * self.q_target.weight / (1e-6 + torch.linalg.norm(self.q_target.weight, dim=1, keepdim=True)))

            scaling = 1 / math.sqrt(self.dimin // self.n_heads)
            self.v_ctx.weight.copy_(scaling * self.v_ctx.weight / (1e-6 + torch.linalg.norm(self.v_ctx.weight, dim=1, keepdim=True)))

            scaling = 1 / math.sqrt(self.dimin)
            self.c_proj.weight.copy_(scaling * self.c_proj.weight / (1e-6 + torch.linalg.norm(self.c_proj.weight, dim=1, keepdim=True)))

    def forward(self, x_ctx, x_target, get_attn_map=False):
        """
        Compute projective attention output 
        """
        # Compute key and value projections from context representations
        k = self.k_ctx(x_ctx)
        v = self.v_ctx(x_ctx)

        # Compute query projection from target representations
        q = self.q_target(x_target)

        # Split into heads
        B, T, _ = x_ctx.size()
        k = k.view(B, T, self.n_heads, self.n_embds // self.n_heads).transpose(1, 2)
        q = q.view(B, T, self.n_heads, self.n_embds // self.n_heads).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.dimin // self.n_heads).transpose(1, 2)

        # Attn map
        if get_attn_map:
            attn_map = get_attention(query=q, key=k)
            torch.cuda.empty_cache()

        # Scaled dot-product attention
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=None,
            dropout_p=0,
            is_causal=True
            )

        # Reshape, project back to original dimension
        d_target = self.c_proj(attn_output.transpose(1, 2).contiguous().view(B, T, self.dimin)) # [batch, length, dimin]

        if get_attn_map:
            return d_target, attn_map
        else:
            return d_target, None


######################## Standard SAE Utils ########################
def rectangle(x):
    # rectangle function
    return ((x >= -0.5) & (x <= 0.5)).float()

class JumpReLU(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, threshold, bandwidth):
        if not isinstance(bandwidth, torch.Tensor):
            bandwidth = torch.tensor(bandwidth, dtype=x.dtype, device=x.device)
        ctx.save_for_backward(x, threshold, bandwidth)
        return x*(x>threshold)

    @staticmethod
    def backward(ctx, grad_output):
        x, threshold, bandwidth = ctx.saved_tensors

        # Compute gradients
        x_grad = (x > threshold).float() * grad_output
        threshold_grad = (
            -(threshold / bandwidth)
            * rectangle((x - threshold) / bandwidth)
            * grad_output
        ).sum(dim=0, keepdim=True)  # Aggregating across batch dimension
        
        return x_grad, threshold_grad, None  # None for bandwidth since const

def jumprelu(x, threshold, bandwidth):
    return JumpReLU.apply(x, threshold, bandwidth)


def softplus_inverse(input, beta=1.0, threshold=20.0):
        """"
        inverse of the softplus function in torch
        """
        if isinstance(input, float):
                input = torch.tensor([input])
        if input*beta<threshold:
                return (1/beta)*torch.log(torch.exp(beta*input)-1.0)
        else:
              return input[0]


class StepFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, threshold, bandwidth):
        if not isinstance(threshold, torch.Tensor):
            threshold = torch.tensor(threshold, dtype=input.dtype, device=input.device)
        if not isinstance(bandwidth, torch.Tensor):
            bandwidth = torch.tensor(bandwidth, dtype=input.dtype, device=input.device)
        ctx.save_for_backward(input, threshold, bandwidth)
        return (input > threshold).type(input.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        x, threshold, bandwidth = ctx.saved_tensors
        grad_input = 0.0*grad_output #no ste to input
        grad_threshold = (
            -(1.0 / bandwidth)
            * rectangle((x - threshold) / bandwidth)
            * grad_output
        ).sum(dim=0, keepdim=True)
        return grad_input, grad_threshold, None  # None for bandwidth since const

def step_fn(input, threshold, bandwidth):
    return StepFunction.apply(input, threshold, bandwidth)