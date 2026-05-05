import pytest
from unittest.mock import patch
import torch
from torch.distributions import MultivariateNormal
from irtorch.irt_dataset import PytorchIRTDataset
from irtorch.estimation_algorithms import MML
from irtorch.models import MonotoneNN


def _make_points_and_log_weights(latent_variables: int, n_points: int = 5, cov: torch.Tensor = None):
    """Helper: build the quasi-MC grid and log-weights for a given dimensionality."""
    if cov is None:
        cov = torch.eye(latent_variables)
    grid = torch.linspace(-3, 3, n_points).view(-1, 1).expand(-1, latent_variables).contiguous()
    if latent_variables > 1:
        combos = torch.cartesian_prod(*[grid[:, i] for i in range(latent_variables)])
    else:
        combos = grid
    log_weights = MultivariateNormal(torch.zeros(latent_variables), cov).log_prob(combos)
    return combos, log_weights


class TestMML:
    @pytest.fixture()
    def irt_model(self, latent_variables, item_categories):
        model = MonotoneNN(
            latent_variables=latent_variables,
            item_categories=item_categories,
            hidden_dim=[3],
        )
        return model

    @pytest.fixture()
    def algorithm(self, data_loaders):
        algorithm = MML()
        algorithm.imputation = "zero"
        algorithm.data_loader = data_loaders
        return algorithm

    # ── _train_step ──────────────────────────────────────────────────────────

    def test__train_step_returns_tuple(self, algorithm: MML, irt_model: MonotoneNN, test_data: torch.Tensor):
        """_train_step must return (float, Tensor) with correct shapes."""
        algorithm.optimizer = torch.optim.Adam(list(irt_model.parameters()), lr=0.01, amsgrad=True)
        points, log_weights = _make_points_and_log_weights(irt_model.latent_variables)
        irt_data = PytorchIRTDataset(test_data)
        loss, ll = algorithm._train_step(irt_model, irt_data, points, log_weights)

        assert isinstance(loss, float)
        n_respondents = test_data.size(0)
        n_points = points.size(0)
        assert ll.shape == (n_points, n_respondents)
        # ll must be detached (no grad)
        assert not ll.requires_grad

    def test__train_step(self, algorithm: MML, irt_model: MonotoneNN, test_data: torch.Tensor):
        """Loss should be non-increasing over two consecutive steps."""
        algorithm.optimizer = torch.optim.Adam(list(irt_model.parameters()), lr=0.01, amsgrad=True)
        points, log_weights = _make_points_and_log_weights(irt_model.latent_variables)
        irt_data = PytorchIRTDataset(test_data)

        previous_loss = float("inf")
        for _ in range(2):
            loss, ll = algorithm._train_step(irt_model, irt_data, points, log_weights)
            assert loss <= previous_loss
            previous_loss = loss

    # ── _evaluate_loss ───────────────────────────────────────────────────────

    def test__evaluate_loss_no_grad(self, algorithm: MML, irt_model: MonotoneNN, test_data: torch.Tensor):
        """_evaluate_loss must not compute gradients and must return a float."""
        algorithm.covariance_matrix = torch.eye(irt_model.latent_variables)
        points, log_weights = _make_points_and_log_weights(irt_model.latent_variables)
        irt_data = PytorchIRTDataset(test_data)

        # Verify no parameters accumulate gradients
        for p in irt_model.parameters():
            p.grad = None

        loss = algorithm._evaluate_loss(irt_model, irt_data, points, log_weights)

        assert isinstance(loss, float)
        assert loss > 0  # negative log-likelihood is positive
        for p in irt_model.parameters():
            assert p.grad is None, "gradients should not accumulate during _evaluate_loss"

    def test__evaluate_loss_consistent_with_train_step(self, algorithm: MML, irt_model: MonotoneNN, test_data: torch.Tensor):
        """When weights are identical, _evaluate_loss and _train_step should report the same value."""
        algorithm.optimizer = torch.optim.Adam(list(irt_model.parameters()), lr=0.0, amsgrad=True)  # lr=0 -> no update
        algorithm.covariance_matrix = torch.eye(irt_model.latent_variables)
        points, log_weights = _make_points_and_log_weights(irt_model.latent_variables)
        irt_data = PytorchIRTDataset(test_data)

        train_loss, _ = algorithm._train_step(irt_model, irt_data, points, log_weights)
        eval_loss = algorithm._evaluate_loss(irt_model, irt_data, points, log_weights)

        assert abs(train_loss - eval_loss) < 1.0  # should be very close (lr=0 means no param update)

    # ── _update_covariance ───────────────────────────────────────────────────

    def test__update_covariance_diagonal_is_one(self):
        """After update, diagonal must always equal 1 (correlation matrix)."""
        alg = MML()
        alg.covariance_matrix = torch.eye(2)
        points, log_weights = _make_points_and_log_weights(2, n_points=7)
        n_resp = 50
        # Synthetic ll: uniform across quadrature points
        ll = torch.zeros(points.size(0), n_resp)

        alg._update_covariance(ll, log_weights, points)

        assert alg.covariance_matrix.shape == (2, 2)
        assert abs(alg.covariance_matrix[0, 0].item() - 1.0) < 1e-5
        assert abs(alg.covariance_matrix[1, 1].item() - 1.0) < 1e-5

    def test__update_covariance_is_positive_definite(self):
        """Resulting covariance must be positive definite (Cholesky succeeds)."""
        alg = MML()
        alg.covariance_matrix = torch.eye(2)
        torch.manual_seed(0)
        points, log_weights = _make_points_and_log_weights(2, n_points=7)
        ll = torch.randn(points.size(0), 100)

        alg._update_covariance(ll, log_weights, points)

        # Cholesky must not raise
        torch.linalg.cholesky(alg.covariance_matrix)

    def test__update_covariance_recovers_correlation(self):
        """
        With a known synthetic posterior (all weight on one correlated point),
        the update should shift the off-diagonal toward that correlation.
        """
        alg = MML()
        alg.covariance_matrix = torch.eye(2)
        points, log_weights = _make_points_and_log_weights(2, n_points=7)
        n_pts, n_resp = points.size(0), 200

        # Make all posterior mass sit on a single point where both dims are positive
        ll = torch.full((n_pts, n_resp), -1e9)
        # Find index closest to (2, 2) — both positive → positive correlation
        target = torch.tensor([2.0, 2.0])
        idx = ((points - target) ** 2).sum(dim=1).argmin().item()
        ll[idx, :] = 0.0

        alg._update_covariance(ll, log_weights, points)

        # Off-diagonal should be positive (and fairly large)
        assert alg.covariance_matrix[0, 1].item() > 0.5

    def test__update_covariance_symmetric(self):
        """Output matrix must be symmetric."""
        alg = MML()
        alg.covariance_matrix = torch.eye(3)
        points, log_weights = _make_points_and_log_weights(3, n_points=5)
        ll = torch.randn(points.size(0), 80)

        alg._update_covariance(ll, log_weights, points)

        assert torch.allclose(alg.covariance_matrix, alg.covariance_matrix.T, atol=1e-6)

    def test__update_covariance_preserves_on_non_pd(self):
        """If jitter cannot restore positive definiteness, the old matrix is kept."""
        alg = MML()
        initial_cov = torch.eye(2)
        alg.covariance_matrix = initial_cov.clone()

        # Force a rank-deficient outer product: all weight on a single zero-mean combination
        # Use only 1 quadrature point so the outer product is rank-1 and thus singular
        points = torch.tensor([[1.0, 1.0]])   # single point
        log_weights = torch.tensor([0.0])
        ll = torch.zeros(1, 10)

        # patch cholesky to always raise so jitter loop exhausts
        original_cholesky = torch.linalg.cholesky
        call_count = {"n": 0}
        def always_fail(m):
            call_count["n"] += 1
            raise torch.linalg.LinAlgError("forced failure")

        with patch("torch.linalg.cholesky", side_effect=always_fail):
            alg._update_covariance(ll, log_weights, points)

        # Should have kept previous covariance
        assert torch.allclose(alg.covariance_matrix, initial_cov)

    # ── fit ──────────────────────────────────────────────────────────────────

    def test_fit(self, algorithm: MML, irt_model: MonotoneNN, test_data):
        """fit() calls _train_step the expected number of times."""
        with patch.object(
            algorithm, "_train_step", return_value=(torch.tensor(0.5), torch.zeros(1, 1))
        ) as mocked_train_step:
            algorithm.fit(
                model=irt_model,
                train_data=test_data[0:100],
                max_epochs=5,
            )
            assert mocked_train_step.call_count == 5

    def test_fit_estimate_covariance_updates_matrix(self, test_data, item_categories):
        """With estimate_covariance=True and 2 LVs, covariance must change from identity."""
        torch.manual_seed(42)
        model = MonotoneNN(latent_variables=2, item_categories=item_categories, hidden_dim=[3])
        alg = MML()
        alg.fit(
            model=model,
            train_data=test_data[0:100],
            max_epochs=20,
            estimate_covariance=True,
            learning_rate_updates_before_stopping=10,
            device="cpu",
        )
        # Diagonal must stay 1
        assert abs(alg.covariance_matrix[0, 0].item() - 1.0) < 1e-4
        assert abs(alg.covariance_matrix[1, 1].item() - 1.0) < 1e-4
        # Must remain positive definite
        torch.linalg.cholesky(alg.covariance_matrix)
        # Must be symmetric
        assert torch.allclose(alg.covariance_matrix, alg.covariance_matrix.T, atol=1e-6)

    def test_fit_estimate_covariance_best_state_restored(self, test_data, item_categories):
        """Best covariance (at best-loss epoch) must be restored after fit."""
        torch.manual_seed(0)
        model = MonotoneNN(latent_variables=2, item_categories=item_categories, hidden_dim=[3])
        alg = MML()
        alg.fit(
            model=model,
            train_data=test_data[0:100],
            max_epochs=15,
            estimate_covariance=True,
            learning_rate_updates_before_stopping=10,
            device="cpu",
        )
        # The stored covariance must at minimum be a valid correlation matrix
        diag = alg.covariance_matrix.diag()
        assert torch.allclose(diag, torch.ones(2), atol=1e-4)
        torch.linalg.cholesky(alg.covariance_matrix)

    def test_fit_estimate_covariance_gauss_hermite_raises(self, test_data, item_categories):
        """estimate_covariance=True with gauss_hermite must raise ValueError."""
        model = MonotoneNN(latent_variables=2, item_categories=item_categories, hidden_dim=[3])
        with pytest.raises(ValueError, match="gauss_hermite"):
            MML().fit(
                model=model,
                train_data=test_data[0:50],
                max_epochs=1,
                integration_method="gauss_hermite",
                estimate_covariance=True,
                device="cpu",
            )

    def test_fit_estimate_covariance_1lv_ignored(self, test_data, item_categories):
        """estimate_covariance=True with 1 LV should be silently ignored (warning logged)."""
        model = MonotoneNN(latent_variables=1, item_categories=item_categories, hidden_dim=[3])
        alg = MML()
        import logging
        with patch.object(
            logging.getLogger("irtorch"), "warning"
        ) as mock_warn:
            alg.fit(
                model=model,
                train_data=test_data[0:50],
                max_epochs=3,
                estimate_covariance=True,
                device="cpu",
            )
            mock_warn.assert_called_once()
            assert "latent variable" in mock_warn.call_args[0][0].lower()

        # Covariance must remain 1x1 identity
        assert alg.covariance_matrix.shape == (1, 1)
        assert abs(alg.covariance_matrix[0, 0].item() - 1.0) < 1e-5

    def test_fit_estimate_covariance_false_leaves_matrix_unchanged(self, test_data, item_categories):
        """estimate_covariance=False must not change the initial covariance matrix."""
        model = MonotoneNN(latent_variables=2, item_categories=item_categories, hidden_dim=[3])
        initial_cov = torch.eye(2)
        alg = MML()
        alg.fit(
            model=model,
            train_data=test_data[0:100],
            max_epochs=5,
            covariance_matrix=initial_cov.clone(),
            estimate_covariance=False,
            device="cpu",
        )
        assert torch.allclose(alg.covariance_matrix, initial_cov)

    # ── _quasi_mc ─────────────────────────────────────────────────────────────

    def test__quasi_mc(self, algorithm: MML):
        algorithm.covariance_matrix = torch.eye(1)
        points, log_weights = algorithm._quasi_mc(5, 1)
        assert points.size() == (5, 1)
        assert log_weights.size() == (5,)

        algorithm.covariance_matrix = torch.eye(2)
        points, log_weights = algorithm._quasi_mc(5, 2)
        assert points.size() == (25, 2)
        assert log_weights.size() == (25,)

    def test__quasi_mc_log_weights_reflect_covariance(self):
        """Log-weights must change when covariance changes."""
        alg = MML()
        alg.covariance_matrix = torch.eye(2)
        _, lw_identity = alg._quasi_mc(5, 2)

        corr = torch.tensor([[1.0, 0.8], [0.8, 1.0]])
        alg.covariance_matrix = corr
        _, lw_corr = alg._quasi_mc(5, 2)

        assert not torch.allclose(lw_identity, lw_corr)
