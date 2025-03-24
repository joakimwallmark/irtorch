# %% 
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from irtorch.torch_modules import CPspline
torch.set_printoptions(precision=10)

# spline = CPspline(deg=1,  n_int=2,  x_range=(0, 1))
# spline = CPspline(deg=3,  n_int=4,  x_range=(0, 1))
# x_fine = np.linspace(0, 1, 500)
# n = 100
# x = np.linspace(0, 1, n)
# y = np.sin(2*np.pi*x) + np.random.normal(0, 0.1, size=n)
# data = pd.DataFrame({'x': x, 'y': y})
# spline.fit(
#     data=data,
#     y_col='y',
#     num_epochs=1000,
#     lr=0.1,
#     constraint_penalty_weight=1.0,
#     smoothing_penalty_weight=0.1
# )

x = torch.linspace(0, 1, 300).view(-1, 1)
x_fine = torch.linspace(0, 1, 300).numpy()
y = torch.sin(2 * np.pi * x)
y = 2*x - 1 
spline = CPspline(deg=3,  n_int=9,  x_range=(0, 1),    int_constraints={
        1: {'+': 0.0}  # First derivative should be positive
    })
spline.knots = spline.generate_knots(0, 1)
spline.theta = torch.nn.Parameter(torch.zeros(12, dtype=torch.float32))
# data = pd.DataFrame({'x': x.flatten().numpy(), 'y': y.flatten().numpy()})
x_fit = x[:299]
y_fit = y[:299]
data = pd.DataFrame({'x': x.flatten().numpy(), 'y': y.flatten().numpy()})
spline.fit(
    data=data,
    y_col='y',
    num_epochs=1000,
    lr=0.1,
    constraint_penalty_weight=10.0,
    smoothing_penalty_weight=1
)
torch.nn.functional.softplus(spline.smoothing_param_unconstrained)

# Evaluate spline derivatives
y_pred = spline.predict(pd.DataFrame({'x': x_fine}))
# print((((y_pred-1)/2)-x_fine).round(5))
y_deriv1 = spline.evaluate_derivative(pd.DataFrame({'x': x_fine}), derivative_order=1)
y_deriv2 = spline.evaluate_derivative(pd.DataFrame({'x': x_fine}), derivative_order=2)
y_deriv3 = spline.evaluate_derivative(pd.DataFrame({'x': x_fine}), derivative_order=3)

# Plot spline and its derivatives
fig, axes = plt.subplots(4, 1, figsize=(12, 16))
titles = ['Spline', '1st Derivative', '2nd Derivative', '3rd Derivative']
y_values = [y_pred, y_deriv1, y_deriv2, y_deriv3]

for ax, title, y_value in zip(axes, titles, y_values):
    ax.plot(x_fine, y_value, label=title)
    ax.set_title(title)
    ax.set_xlabel('x')
    ax.set_ylabel('Value')
    ax.legend()
    ax.grid(True)

plt.tight_layout()
plt.show()

# %%
torch.set_printoptions(precision=4)
(torch.tensor(spline.predict(data["x"]))-y).round(decimals=4).flatten()


# %% Test different constraint types
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from irtorch.torch_modules import CPspline

# Generate sample data
np.random.seed(42)
n = 200
x = np.linspace(0, 1, n)
y = np.sin(2*np.pi*x) + 0.5*x + np.random.normal(0, 0.1, size=n)
data = pd.DataFrame({'x': x, 'y': y})

# plot data
plt.figure(figsize=(10, 6))
plt.scatter(x, y, alpha=0.5, label='Data points')   
plt.xlabel('x')
plt.ylabel('y')
plt.show()

# Test 1: Monotone increasing constraint
spline_monotone = CPspline(
    deg=3,
    n_int=20,
    x_range=(0, 1),
    int_constraints={
        1: {'+': 0.0}  # First derivative should be positive
    }
)

# Test 2: Monotone decreasing constraint
spline_decreasing = CPspline(
    deg=3,
    n_int=20,
    x_range=(0, 1),
    int_constraints={
        1: {'-': 0.0}  # First derivative should be negative
    }
)

# Test 3: Point constraints
point_data = pd.DataFrame({
    'x': [0.2, 0.5, 0.8],
    'y': [0.3, 0.6, 0.9]
})
spline_points = CPspline(
    deg=3,
    n_int=20,
    x_range=(0, 1),
    pt_constraints={
        0: {'equalsTo': point_data}  # Function should pass through these points
    }
)

# Test 4: Combined constraints (monotone + point)
spline_combined = CPspline(
    deg=3,
    n_int=20,
    x_range=(0, 1),
    int_constraints={
        1: {'+': 0.0}  # Monotone increasing
    },
    pt_constraints={
        0: {'greaterThan': point_data}  # Function should be above these points
    }
)

# Test 5: PDF constraint (non-negative and integrates to 1)
spline_pdf = CPspline(
    deg=3,
    n_int=20,
    x_range=(0, 1),
    pdf_constraint=True
)

# Fit all splines
splines = {
    'Monotone Increasing': spline_monotone,
    # 'Monotone Decreasing': spline_decreasing,
    # 'Point Constraints': spline_points,
    # 'Combined Constraints': spline_combined,
    'PDF Constraint': spline_pdf
}

# Fit and plot all splines
fig, axes = plt.subplots(len(splines), 1, figsize=(12, 4*len(splines)))
x_fine = np.linspace(0, 1, 200)
data_fine = pd.DataFrame({'x': x_fine})

for idx, (name, spline) in enumerate(splines.items()):
    # Adjust parameters for PDF constraint
    if name == 'PDF Constraint':
        constraint_weight = 50.0  # Increased from 10.0
        num_epochs = 10000       # Increased from 5000
        learning_rate = 0.1      # Adjusted
    else:
        constraint_weight = 1.0
        num_epochs = 2000
        learning_rate = 0.1
    
    # Fit the spline
    spline.fit(
        data=data,
        y_col='y',
        num_epochs=num_epochs,
        lr=learning_rate,
        constraint_penalty_weight=constraint_weight,
        smoothing_penalty_weight=0.1,
        early_stopping_patience=300  # Increased from 200
    )
    
    # Generate predictions
    y_pred = spline.predict(data_fine)
    
    # Plot
    ax = axes[idx]
    ax.scatter(x, y, alpha=0.3, label='Data points', color='blue')
    ax.plot(x_fine, y_pred, 'r-', label='Fitted spline', linewidth=2)
    
    if name == 'PDF Constraint':
        # Add integral value to title
        integral = np.trapz(y_pred, x_fine)
        ax.set_title(f'{name} (Integral = {integral:.3f})')
        # Force y-axis to start at 0 for PDF
        ax.set_ylim(bottom=0)
    else:
        ax.set_title(name)
    
    # For point constraints, plot the constraint points
    if 'pt_constraints' in name.lower():
        ax.scatter(point_data['x'], point_data['y'], 
                  color='green', s=100, label='Constraint points')
    
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.legend()
    ax.grid(True)

plt.tight_layout()
plt.show()

# Verification code
print("\nPDF Constraint Verification:")
print("-" * 30)
x_verify = np.linspace(0, 1, 10000)  # Increased resolution
data_verify = pd.DataFrame({'x': x_verify})
y_verify = splines['PDF Constraint'].predict(data_verify)

integral = np.trapz(y_verify, x_verify)
min_value = np.min(y_verify)
print(f"Integral of PDF: {integral:.6f}")
print(f"Minimum value: {min_value:.6f}")

if abs(integral - 1.0) < 5e-3 and min_value >= -1e-6:  # Slightly more lenient tolerance
    print("✓ PDF constraints satisfied")
else:
    print("✗ PDF constraints not satisfied")
    print(f"Deviation from 1.0: {abs(integral - 1.0):.6f}")

# %%










# %%
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from irtorch.torch_modules import CPspline

# Generate non-monotone data (sine wave with upward trend)
np.random.seed(42)
n=500
x = np.linspace(0, 1, n)
# Combining sine wave with linear trend and noise
y = 2*np.sin(8*x) + 8*x + np.random.normal(0, 0.2, size=n)

# Create a pandas DataFrame
data = pd.DataFrame({
    'x': x,
    'y': y
})

# Plot original data
plt.figure(figsize=(10, 6))
plt.scatter(x, y, alpha=0.5, label='Original data')
plt.title('Original Non-monotone Data')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()

# %%
# Initialize the CPspline with monotonicity constraint
# We'll use int_constraints to enforce positive first derivative (monotone increasing)
spline = CPspline(
    deg=3,  # cubic spline
    n_int=20,  # 20 intervals
    x_range=(0, 1),
    int_constraints={
        1: {'+': 0.0}  # First derivative should be positive (monotone increasing)
    }
)

# %%
# Fit the spline
spline.fit(
    data=data,
    y_col='y',
    num_epochs=4200,
    lr=0.1,
    constraint_penalty_weight=1.0,
    smoothing_penalty_weight=0.1,
    early_stopping_patience=150,
    early_stopping_tol=1e-10
)

# %%
# Generate predictions on a fine grid for plotting
x_fine = np.linspace(0, 1, 200)
data_fine = pd.DataFrame({'x': x_fine})
y_pred = spline.predict(data_fine)

# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

# Plot the fitted spline
ax1.scatter(x, y, alpha=0.5, label='Data points')
ax1.plot(x_fine, y_pred, 'r-', label='Fitted spline')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_title('Monotone Spline Fit')
ax1.legend()
ax1.grid(True)

# Plot the first derivative
y_deriv = spline.evaluate_derivative(data_fine, derivative_order=1)
ax2.plot(x_fine, y_deriv, 'b-', label='First derivative')
ax2.axhline(y=0, color='k', linestyle='--', alpha=0.3)
ax2.set_xlabel('x')
ax2.set_ylabel("y'")
ax2.set_title('First Derivative (should be positive)')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.show()

# Print some metrics
mse = np.mean((spline.predict(data.iloc[:, 0]) - y) ** 2)
print(f"Mean Squared Error: {mse:.6f}")

# %%
# Verify monotonicity
print(f"negative derivatives locations: {x_fine[y_deriv < 0]}")
min_derivative = np.min(y_deriv)
print(f"\nMinimum value of first derivative: {min_derivative:.6f}")

# %%
# Now let's try different constraint weights to show their effect
constraint_weights = [0.1, 1.0, 10.0]
fig, axes = plt.subplots(len(constraint_weights), 1, figsize=(12, 4*len(constraint_weights)))

for idx, weight in enumerate(constraint_weights):
    # Initialize and fit the spline
    spline = CPspline(
        deg=3,
        n_int=20,
        x_range=(0, 1.01),
        int_constraints={
            1: {'+': 0.0}  # Force positive first derivative
        }
    )
    
    spline.fit(
        data=data,
        y_col='y',
        num_epochs=10000,
        lr=0.05,
        constraint_penalty_weight=weight,
        smoothing_penalty_weight=0.1,
        early_stopping_patience=300,
        early_stopping_tol=1e-10
    )
    
    # Generate predictions
    y_pred = spline.predict(data_fine)
    mse = np.mean((spline.predict(data.iloc[:, 0]) - y) ** 2)
    # Verify monotonicity
    y_deriv = spline.evaluate_derivative(data.iloc[:, 0], derivative_order=1)
    print(f"negative derivatives locations: {x[y_deriv < 0]}")
    min_derivative = np.min(y_deriv)

    # Plot results
    ax = axes[idx]
    ax.scatter(x, y, alpha=0.3, label='Data points', color='blue')
    ax.plot(x_fine, y_pred, 'r-', label='Fitted spline', linewidth=2)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(f'Monotone Spline Fit (constraint weight = {weight}, MSE = {mse:.6f}, minimum derivative = {min_derivative:.6f})')
    ax.legend()
    ax.grid(True)
    


plt.tight_layout()
print("(Should be positive for monotone increasing function)")

# %%
spline = CPspline(
    deg=3,  # cubic spline
    n_int=20,  # 20 intervals
    x_range=(0, 1),
    int_constraints={
        1: {'+': 0.0}  # First derivative should be positive (monotone increasing)
    }
)


spline.theta[2] =
spline.theta
spline.knots
pd.DataFrame('x': 9.82)

spline.predict(data_fine)
spline.predict(data_fine)

# %%
import numpy as np
import matplotlib.pyplot as plt
x = np.linspace(0, 1, 100)
knots = np.linspace(0, 1, 10)
deg = 3
B = spline.bspline_basis(x, knots, deg)
for i in range(B.shape[1]):
    plt.plot(x, B[:, i], label=f'Basis {i}')
for knot in knots:
    plt.axvline(x=knot, color='k', linestyle='--', alpha=0.3)

plt.legend()
plt.xlabel('x')
plt.ylabel('Basis function value')
plt.title('B-spline Basis Functions')
plt.show()
# %%
def bspline_basis(x, knots, deg):
    n_bases = len(knots) - deg - 1
    B = np.zeros((len(x), n_bases))
    for i in range(n_bases):
        B[:, i] = bspline_basis_function(x, knots, deg, i)
    return B

def bspline_basis_function(x, knots, deg, i):
    if deg == 0:
        return np.where((knots[i] <= x) & (x < knots[i+1]), 1.0, 0.0)
    else:
        denom1 = knots[i+deg] - knots[i]
        denom2 = knots[i+deg+1] - knots[i+1]
        coef1 = 0.0
        coef2 = 0.0
        if denom1 > 0:
            coef1 = (x - knots[i]) / denom1
        if denom2 > 0:
            coef2 = (knots[i+deg+1] - x) / denom2
        return coef1 * bspline_basis_function(x, knots, deg-1, i) + \
                coef2 * bspline_basis_function(x, knots, deg-1, i+1)

def predict(data, knots, deg, theta: torch.Tensor) -> np.ndarray:
    if data.empty:
        return np.array([])

    if isinstance(data, pd.Series):
        x_new = data.values.flatten()
    elif isinstance(data, pd.DataFrame):
        x_new = data.values.flatten()
    else:
        raise ValueError("Input data must be a pandas Series or DataFrame.")

    B_new = bspline_basis(x_new, knots, deg)
    B_new = torch.tensor(B_new, dtype=torch.float32)

    with torch.no_grad():
        y_pred = B_new @ theta
    return y_pred.numpy()

# %%
import torch
import numpy as np
import matplotlib.pyplot as plt
x = np.linspace(0, 1, 100)
deg = 3
knots = np.linspace(0, 1, 11)
knots = np.concatenate((np.zeros(deg), knots, np.ones(deg)))
basis = bspline_basis(x, knots, deg)

for i in range(basis.shape[1]):
    plt.plot(x, basis[:, i], label=f'Basis {i}')
for knot in knots:
    plt.axvline(x=knot, color='k', linestyle='--', alpha=0.3)

plt.legend()
plt.xlabel('x')
plt.ylabel('Basis function value')
plt.title('B-spline Basis Functions')
plt.show()

# %%
basis_new = bspline_basis(torch.arange(0, 1, 0.1), knots, deg)
theta = torch.arange(1, 1 + basis.shape[1], dtype=torch.float32)
y_pred = torch.tensor(basis_new, dtype=torch.float32) @ theta











# %%
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from irtorch.torch_modules import CPspline
knots = torch.tensor([0, 0, 0, 0, 0.2, 0.4, 0.6, 0.8, 1, 1, 1, 1])
degree = 3
spline = CPspline(
    deg=3,  # cubic spline
    n_int=5,  # 20 intervals
    x_range=(0, 1),
    # int_constraints={
    #     1: {'+': 0.0}  # First derivative should be positive (monotone increasing)
    # }
)
spline.fit(
    data=data,
    y_col='y',
    num_epochs=4200,
    lr=0.1,
    constraint_penalty_weight=1.0,
    smoothing_penalty_weight=0.1,
    early_stopping_patience=150,
    early_stopping_tol=1e-10
)

x_fine = np.linspace(-1, 0.2, 200)
data_fine = pd.DataFrame({'x': x_fine})
y_pred = spline.predict(data_fine)

# %%
x_fine = np.linspace(0, 1, 500)
x_fine = np.concatenate((x_fine, knots.numpy()))
x_fine.sort()
# x_fine = torch.cat((torch.tensor(x_fine), knots))
# x_fine, _ = torch.sort(x_fine)
basis = spline.bspline_basis(x_fine, knots.numpy(), degree)
basis_deriv = spline.bspline_basis_derivative(spline.theta.shape[0], 1, x=x_fine)
basis_deriv2 = spline.bspline_basis_derivative(spline.theta.shape[0], 2, x=x_fine)
basis_deriv3 =spline.bspline_basis_derivative(spline.theta.shape[0], 3, x=x_fine)

# plot basis functions and derivatives
fig, axes = plt.subplots(4, 1, figsize=(10, 10))
for i in range(basis.shape[1]):
    axes[0].plot(x_fine, basis[:, i], label=f'Basis {i}')
    axes[1].plot(x_fine, basis_deriv[:, i], label=f'Derivative {i}')
    axes[2].plot(x_fine, basis_deriv2[:, i], label=f'Derivative {i}')
    axes[3].plot(x_fine, basis_deriv3[:, i], label=f'Derivative {i}')
