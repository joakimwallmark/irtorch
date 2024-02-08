import torch
from torch.autograd import Function

# TODO REMOVE IF NOT NEEDED (use -elu(-x) directly)
class ConvexELU(Function):
    @staticmethod
    def forward(ctx, input, alpha):
        ctx.save_for_backward(input)
        ctx.alpha = alpha
        return torch.where(input < 0, input, - alpha * (torch.exp(-input) - 1))

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        alpha = ctx.alpha
        grad_input = grad_output.clone()
        # We multiply the upstream gradient with the gradient of out activation function (chain rule)
        grad_input[input >= 0] *= alpha * torch.exp(-input[input >= 0])
        # For input < 0, the gradient is already correctly set to grad_output, so no change needed
        return grad_input, None

class BoundedELU(Function):
    @staticmethod
    def forward(ctx, input, alpha=1.0):
        ctx.save_for_backward(input)
        ctx.alpha = alpha

        output = input.clone()
        # Implementing the piecewise function
        mask1 = input <= -1
        mask2 = input >= 1
        output[mask1] = (alpha * (torch.exp(input[mask1] + 1) - 1) - 1).to(output.dtype)
        output[mask2] = (-alpha * (torch.exp(-input[mask2] + 1) - 1) + 1).to(output.dtype)
        # No change for -1 < input < 1 as output is same as input

        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        alpha = ctx.alpha

        grad_input = grad_output.clone()
        mask1 = input <= -1
        mask2 = input >= 1
        grad_input[mask1] = alpha * torch.exp(input[mask1] + 1) * grad_output[mask1]
        grad_input[mask2] = alpha * torch.exp(-input[mask2] + 1) * grad_output[mask2]
        # No change for -1 < input < 1 as gradient is 1

        return grad_input, None