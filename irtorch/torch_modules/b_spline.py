import torch
from splinetorch.b_spline_basis import b_spline_basis, b_spline_basis_derivative

class BSplineBasisFunction(torch.autograd.Function):
    generate_vmap_rule = True

    @staticmethod
    def setup_context(ctx, inputs, output):
        # inputs is a tuple: (x, knots, degree)
        x, knots, degree = inputs
        # Save tensors needed for backward. Non-tensor inputs can be stored directly on ctx.
        ctx.save_for_backward(x, knots)
        ctx.degree = degree

    @staticmethod
    def forward(x, knots, degree):
        basis = b_spline_basis(x, knots, degree)
        return basis

    @staticmethod
    def backward(ctx, grad_output):
        # Retrieve saved tensors and attributes
        x, knots = ctx.saved_tensors
        degree = ctx.degree

        # Check if gradient calculation is needed for x
        # grad_output has shape (*batch_dims, n_points, n_bases) when vmapped
        # d_basis_dx has shape (n_points, n_bases) or potentially batched if inputs were batched originally.
        # Ensure dimensions align for broadcasting or use einsum/matmul carefully.

        grad_x = None
        if ctx.needs_input_grad[0]: # Check if gradient wrt x is needed
            # Compute the analytic derivative of the B-spline basis with respect to x.
            d_basis_dx = b_spline_basis_derivative(x, knots, degree, order=1)

            # Calculate the gradient for x. Need to handle potential batch dimensions added by vmap.
            # Original logic: grad_x = (grad_output * d_basis_dx).sum(dim=-1)
            # This assumes grad_output and d_basis_dx have compatible shapes.
            # Let's make it more robust using einsum if shapes might vary due to vmap.
            # Assume grad_output is (*batch, points, bases) and d_basis_dx is (points, bases)
            # We want grad_x to be (*batch, points)

            # If d_basis_dx is not batched, unsqueeze it for broadcasting
            if d_basis_dx.dim() < grad_output.dim():
                # Add batch dimensions to match grad_output
                num_batch_dims = grad_output.dim() - d_basis_dx.dim()
                d_basis_dx_expanded = d_basis_dx.view((1,) * num_batch_dims + d_basis_dx.shape)
            else:
                d_basis_dx_expanded = d_basis_dx

            # Element-wise product and sum over the 'bases' dimension
            grad_x = (grad_output * d_basis_dx_expanded).sum(dim=-1)


        # Gradients for knots and degree are None because they are not tensors requiring grad
        # or we've explicitly decided not to compute them.
        # The number of returned gradients must match the number of inputs to forward.
        return grad_x, None, None
