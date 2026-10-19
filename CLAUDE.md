# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

IRTorch is a Python package for item response theory (IRT) built on PyTorch. It is published to PyPI as `irtorch`, and its docs are on Read the Docs.

## Commands

The project uses `uv` with the `dev` dependency group. Torch is pinned to the CUDA 12.6 wheel index in `pyproject.toml`.

```bash
uv sync --group dev                       # install (same as CI)
uv run pytest                             # full suite (CI also adds --cov=irtorch --cov-report=xml)
uv run pytest -n auto                     # parallel via pytest-xdist (~35 s)
uv run pytest tests/test_mml.py           # one file
uv run pytest tests/test_mml.py::TestMML::test_name   # one test
uv run pytest -m integration              # only integration-marked tests (--strict-markers is on)
cd docs && make html                      # Sphinx docs (autodoc + napoleon, nbsphinx notebooks)
```

Run pytest from the repo root, because `tests/conftest.py` loads `tests/datasets/test_data.pt` by a relative path. CI runs on Python 3.11 and 3.12. The package version is set in two places, `pyproject.toml` and `irtorch/__init__.py`, so update both when bumping it. There is no linter configured.

## Architecture

The user-facing workflow is to construct a model, call `model.fit(train_data, algorithm=SomeAlgorithm())`, and then use `model.latent_scores(...)`, `model.evaluate.*`, `model.plot.*`, and optionally `model.add_scale_transformation(...)`.

**Models** (`irtorch/models/`): every model subclasses `BaseIRTModel` (`nn.Module` + ABC). The contract is:
- `forward(theta)` takes `theta` of shape `(respondents, latent_variables)` and returns logits of shape `(respondents, items * max_item_responses)`. Each item gets a fixed-width block of `max_item_responses` columns. Categories an item doesn't have are set to `-inf` logits (for example `self.missing_category`).
- `probabilities_from_output` reshapes these logits to `(respondents, items, max_item_responses)`. The default is a softmax. Models whose probabilities aren't a softmax (GRM, 3PL, NestedLogit, SurprisalSpline, ModelMix) override both this method and `log_likelihood`.
- Most constructors accept either `item_categories` or `data`. With `data`, the category counts are inferred from the per-column max.
- `mc_correct` (the correct category per multiple-choice item) is validated in the base class and used by models like `NestedLogit`.
- `ModelMix` combines several models that share one set of latent variables. Its items are the concatenation of the sub-models' items, in order.
- `save_model`/`load_model` store the state dict together with the fitted `algorithm` and the `scale` list.

**Estimation algorithms** (`irtorch/estimation_algorithms/`): `MML`, `JML`, `AE`, and `VAE` subclass `BaseIRTAlgorithm`. `model.fit` delegates to `algorithm.fit(model, ...)` and then stores the algorithm on `model.algorithm`. Downstream code depends on attributes the algorithm sets during fitting:
- `train_data`
- `training_theta_scores`
- `training_history`
- `covariance_matrix` (MML)

`Evaluator`, `Plotter`, and EAP/ML scoring read these through `model.algorithm`, so a new algorithm must populate them. `AE` and `VAE` use encoder networks from `estimation_algorithms/encoders/`.

**Scales** (`irtorch/rescale/`): a `Scale` subclass (`Bit`, `Flow`, `RankCDF`, `Rotate`, `Reverse`, `LinkCommonItems`) implements `transform`, an optional `inverse`, and `jacobian`. Models keep a list in `model.scale`, and `transform_theta` / `theta_transform_jacobian` chain the list in order. Gradient and information methods take a `rescale` flag that applies the Jacobian chain.

**Evaluator / Plotter** (`evaluator.py`, `plotter.py`): these are wrappers created lazily from the `model.evaluate` and `model.plot` properties. They hold a reference to the model and get fit state from `model.algorithm`. The plots use Plotly.

**Missing data**: responses are coded `0, 1, …`. Missing values may be `NaN` or `-1`. Use `_internal_utils.get_missing_mask` to detect them rather than checking for only one of the two. Imputation helpers live in `_internal_utils.py`.

**Device convention**: models live on the CPU. Each fit algorithm, and the ML/MAP scoring path, takes a `device` argument, moves to that device temporarily, and ends with `model.to("cpu")`. Code that runs on a model the user placed on CUDA themselves (for example evaluator methods or EAP grids) is not expected to work. Don't treat device mismatches there as bugs. The `device` fixture in `conftest.py` parametrizes tests over cpu/cuda when CUDA is available, and it belongs only on functions that take a `device` argument.

**Other modules**:
- `load_dataset.py`: bundled datasets, stored as `.pt` files under `irtorch/datasets/` and listed in `package-data`.
- `torch_modules/`: reusable layers (splines, monotone polynomial, GMM, normalizing flow) used by models and scales.
- `utils.py`: public helpers re-exported from `irtorch`.
- `config.py`: `set_verbosity`. The package logs through the `"irtorch"` logger.

## Conventions

- Docstrings are NumPy style because Sphinx autodoc renders them. Public classes are listed explicitly in `docs/source/*.rst`, so add new public classes there.
- Use `from __future__ import annotations` with `TYPE_CHECKING` imports to avoid circular imports between models, algorithms, and scales. `Evaluator` and `Plotter` are imported lazily inside the properties for the same reason.
