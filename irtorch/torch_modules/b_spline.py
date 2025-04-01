import torch
from splinetorch.b_spline_basis import b_spline_basis, b_spline_basis_derivative

class BSplineBasisFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, knots, degree):
        # Save inputs needed for backward.
        ctx.save_for_backward(x, knots)
        ctx.degree = degree
        # Compute the forward B-spline basis using your original function.
        # (This is the non-smooth version, so autograd's default backward would be zero.)
        basis = b_spline_basis(x, knots, degree)
        return basis

    @staticmethod
    def backward(ctx, grad_output):
        x, knots = ctx.saved_tensors
        degree = ctx.degree
        # Compute the analytic derivative of the B-spline basis with respect to x.
        d_basis_dx = b_spline_basis_derivative(x, knots, degree, order=1)
        # grad_output has shape (n_points, n_bases). For each x[i], the contribution is:
        # dLoss/dx[i] = sum_j grad_output[i,j] * d_basis_dx[i,j]
        grad_x = (grad_output * d_basis_dx).sum(dim=1)
        # We assume no gradients are needed with respect to knots or degree.
        return grad_x, None, None