import logging
import torch

logger = logging.getLogger('irtorch')

class BoundedELU(torch.autograd.Function):
    # Set generate_vmap_rule to True to ask PyTorch to automatically generate a vmap rule.
    generate_vmap_rule = True

    @staticmethod
    def forward(input_tensor: torch.Tensor, alpha=1.0):
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
        input_tensor : torch.Tensor
            The input tensor.
        alpha : float, optional
            The alpha parameter. Default is 1.0.

        Returns
        -------
        torch.Tensor
            The output tensor.
        """
        neg_mask = (input_tensor <= -1).float()
        pos_mask = (input_tensor >= 1).float()
        middle_mask = 1 - neg_mask - pos_mask

        neg_part = alpha * (torch.exp(input_tensor + 1) - 1) - 1
        pos_part = -alpha * (torch.exp(-input_tensor + 1) - 1) + 1
        middle_part = input_tensor

        # we need to remove possible infinities to avoid nan in the output
        neg_part = torch.where(neg_part == float('inf'), torch.zeros_like(neg_part), neg_part)
        pos_part = torch.where(pos_part == float('-inf'), torch.zeros_like(pos_part), pos_part)

        output = neg_mask * neg_part + pos_mask * pos_part + middle_mask * middle_part

        return output

    @staticmethod
    def setup_context(ctx, inputs, output):
        # Method needed for functorch compatibility (vmap etc.)
        # Setup any necessary context for functorch transforms
        input_tensor, alpha = inputs
        ctx.save_for_backward(input_tensor)
        ctx.alpha = alpha

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
        input_tensor, = ctx.saved_tensors
        alpha = ctx.alpha

        neg_mask = (input_tensor <= -1).float()
        pos_mask = (input_tensor >= 1).float()
        middle_mask = 1 - neg_mask - pos_mask

        neg_part_grad = alpha * torch.exp(input_tensor + 1)
        pos_part_grad = alpha * torch.exp(-input_tensor + 1)
        middle_part_grad = 1

        # Ensuring gradient is not infinite
        neg_part_grad = torch.where(neg_part_grad == float('inf'), torch.zeros_like(neg_part_grad), neg_part_grad)
        pos_part_grad = torch.where(pos_part_grad == float('inf'), torch.zeros_like(pos_part_grad), pos_part_grad)

        grad_input = neg_mask * neg_part_grad + pos_mask * pos_part_grad + middle_mask * middle_part_grad
        return grad_input * grad_output, None
