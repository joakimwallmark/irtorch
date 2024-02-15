import logging
import torch
from torch.autograd import Function

logger = logging.getLogger(__name__)

class BoundedELU(Function):
    @staticmethod
    def forward(ctx, input, alpha=1.0):
        """
        Applies the bounded ELU function element-wise to the input tensor.

        The bounded ELU function is defined as:

        .. math::
            f(x) = \\begin{cases}
            \\alpha (e^x - 1) & \\text{if } x \\leq 0 \\\\
            x & \\text{if } 0 < x < 1 \\\\
            -\\alpha (e^{-x} - 1) & \\text{if } x \\geq 1
            \\end{cases}

        Parameters
        ----------
        input : torch.Tensor
            The input tensor.
        alpha : float, optional
            The alpha parameter. Default is 1.0.

        Returns
        -------
        torch.Tensor
            The output tensor.
        """
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
        """
        Computes the gradient of the bounded ELU function.

        Parameters
        ----------
        grad_output : torch.Tensor
            The gradient of the loss with respect to the output of the bounded ELU function.

        Returns
        -------
        torch.Tensor
            The gradient of the loss with respect to the input of the bounded ELU function.
        """
        input, = ctx.saved_tensors
        alpha = ctx.alpha

        grad_input = grad_output.clone()
        mask1 = input <= -1
        mask2 = input >= 1
        grad_input[mask1] = alpha * torch.exp(input[mask1] + 1) * grad_output[mask1]
        grad_input[mask2] = alpha * torch.exp(-input[mask2] + 1) * grad_output[mask2]
        # No change for -1 < input < 1 as gradient is 1

        return grad_input, None