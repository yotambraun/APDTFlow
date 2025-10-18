"""
Comprehensive test suite for exogenous variables functionality.
Tests the ExogenousFeatureFusion module and integration with APDTFlow.
"""
import torch
import numpy as np
import pytest
from apdtflow.exogenous import ExogenousFeatureFusion, ExogenousProcessor


class TestExogenousFeatureFusion:
    """Test suite for ExogenousFeatureFusion module."""

    def test_concat_fusion(self):
        """Test concatenation-based fusion."""
        batch_size = 4
        seq_len = 30
        num_exog = 3
        hidden_dim = 16

        fusion = ExogenousFeatureFusion(
            hidden_dim=hidden_dim,
            num_exog_features=num_exog,
            fusion_type='concat'
        )

        # Create dummy data
        target = torch.randn(batch_size, 1, seq_len)
        exog = torch.randn(batch_size, num_exog, seq_len)

        # Forward pass
        output = fusion(target, exog)

        # Check output shape
        assert output.shape == (batch_size, 1, seq_len)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_gated_fusion(self):
        """Test gated fusion (recommended approach)."""
        batch_size = 4
        seq_len = 30
        num_exog = 3
        hidden_dim = 16

        fusion = ExogenousFeatureFusion(
            hidden_dim=hidden_dim,
            num_exog_features=num_exog,
            fusion_type='gated'
        )

        target = torch.randn(batch_size, 1, seq_len)
        exog = torch.randn(batch_size, num_exog, seq_len)

        output = fusion(target, exog)

        assert output.shape == (batch_size, 1, seq_len)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

        # Test that gate values are between 0 and 1
        with torch.no_grad():
            target_encoded = fusion.target_encoder(target.transpose(1, 2))
            exog_encoded = fusion.exog_encoder(exog.transpose(1, 2))
            gate_input = torch.cat([target_encoded, exog_encoded], dim=-1)
            gate = fusion.gate(gate_input)
            assert torch.all(gate >= 0) and torch.all(gate <= 1)

    def test_attention_fusion(self):
        """Test attention-based fusion."""
        batch_size = 4
        seq_len = 30
        num_exog = 3
        hidden_dim = 16

        fusion = ExogenousFeatureFusion(
            hidden_dim=hidden_dim,
            num_exog_features=num_exog,
            fusion_type='attention'
        )

        target = torch.randn(batch_size, 1, seq_len)
        exog = torch.randn(batch_size, num_exog, seq_len)

        output = fusion(target, exog)

        assert output.shape == (batch_size, 1, seq_len)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_invalid_fusion_type(self):
        """Test that invalid fusion type raises error."""
        with pytest.raises(ValueError):
            ExogenousFeatureFusion(
                hidden_dim=16,
                num_exog_features=3,
                fusion_type='invalid'
            )

    def test_different_batch_sizes(self):
        """Test with various batch sizes."""
        for batch_size in [1, 2, 8, 16]:
            fusion = ExogenousFeatureFusion(
                hidden_dim=16,
                num_exog_features=3,
                fusion_type='gated'
            )

            target = torch.randn(batch_size, 1, 30)
            exog = torch.randn(batch_size, 3, 30)

            output = fusion(target, exog)
            assert output.shape == (batch_size, 1, 30)

    def test_different_sequence_lengths(self):
        """Test with various sequence lengths."""
        for seq_len in [10, 20, 50, 100]:
            fusion = ExogenousFeatureFusion(
                hidden_dim=16,
                num_exog_features=3,
                fusion_type='gated'
            )

            target = torch.randn(4, 1, seq_len)
            exog = torch.randn(4, 3, seq_len)

            output = fusion(target, exog)
            assert output.shape == (4, 1, seq_len)

    def test_gradient_flow(self):
        """Test that gradients flow properly through fusion."""
        fusion = ExogenousFeatureFusion(
            hidden_dim=16,
            num_exog_features=3,
            fusion_type='gated'
        )

        target = torch.randn(4, 1, 30, requires_grad=True)
        exog = torch.randn(4, 3, 30, requires_grad=True)

        output = fusion(target, exog)
        loss = output.sum()
        loss.backward()

        # Check that gradients exist
        assert target.grad is not None
        assert exog.grad is not None
        assert not torch.isnan(target.grad).any()
        assert not torch.isnan(exog.grad).any()


class TestExogenousProcessor:
    """Test suite for ExogenousProcessor utility."""

    def test_validate_exog_data_valid(self):
        """Test validation with valid data."""
        past_exog = np.random.randn(100, 3)
        future_exog = np.random.randn(20, 3)  # Same number of features
        forecast_horizon = 10

        # Should not raise
        ExogenousProcessor.validate_exog_data(past_exog, future_exog, forecast_horizon)

    def test_validate_exog_data_invalid_future_length(self):
        """Test validation with insufficient future data."""
        past_exog = np.random.randn(100, 3)
        future_exog = np.random.randn(5, 2)  # Only 5 samples
        forecast_horizon = 10  # Need 10

        with pytest.raises(ValueError, match="future_exog must have at least"):
            ExogenousProcessor.validate_exog_data(past_exog, future_exog, forecast_horizon)

    def test_normalize_exog(self):
        """Test normalization of exogenous features."""
        exog = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

        exog_norm, mean, std = ExogenousProcessor.normalize_exog(exog)

        # Check normalization
        assert exog_norm.shape == exog.shape
        assert np.allclose(np.mean(exog_norm, axis=0), 0, atol=1e-7)
        assert np.allclose(np.std(exog_norm, axis=0), 1, atol=1e-7)

        # Check that we can denormalize
        exog_denorm = exog_norm * std + mean
        assert np.allclose(exog_denorm, exog)

    def test_normalize_exog_constant_feature(self):
        """Test normalization with constant feature."""
        exog = np.array([[1.0, 5.0, 3.0], [1.0, 6.0, 6.0], [1.0, 7.0, 9.0]])

        exog_norm, mean, std = ExogenousProcessor.normalize_exog(exog)

        # Constant feature should have std = 1.0 (set to avoid division by zero)
        assert std[0, 0] == 1.0
        # Constant feature should remain constant after normalization
        assert np.allclose(exog_norm[:, 0], exog_norm[0, 0])

    def test_denormalize_exog(self):
        """Test denormalization using manual computation."""
        exog = np.random.randn(100, 3) * 10 + 50
        exog_norm, mean, std = ExogenousProcessor.normalize_exog(exog)

        # Manually denormalize
        exog_denorm = exog_norm * std + mean

        assert np.allclose(exog_denorm, exog, rtol=1e-5)


class TestAPDTFlowWithExogenous:
    """Test APDTFlow model integration with exogenous variables."""

    def test_apdtflow_with_exog_forward(self):
        """Test APDTFlow forward pass with exogenous variables."""
        from apdtflow.models.apdtflow import APDTFlow

        batch_size = 4
        T_in = 30
        num_exog = 3

        model = APDTFlow(
            num_scales=3,
            input_channels=1,
            filter_size=5,
            hidden_dim=16,
            output_dim=1,
            forecast_horizon=7,
            use_embedding=True,
            num_exog_features=num_exog,
            exog_fusion_type='gated'
        )

        target = torch.randn(batch_size, 1, T_in)
        exog = torch.randn(batch_size, num_exog, T_in)
        t_span = torch.linspace(0, 1, steps=T_in)

        preds, pred_logvars = model(target, t_span, exog=exog)

        assert preds.shape == (batch_size, 7, 1)
        assert pred_logvars.shape == (batch_size, 7, 1)
        assert not torch.isnan(preds).any()
        assert not torch.isnan(pred_logvars).any()

    def test_apdtflow_without_exog(self):
        """Test that model works without exog when num_exog_features=0."""
        from apdtflow.models.apdtflow import APDTFlow

        batch_size = 4
        T_in = 30

        model = APDTFlow(
            num_scales=3,
            input_channels=1,
            filter_size=5,
            hidden_dim=16,
            output_dim=1,
            forecast_horizon=7,
            use_embedding=True,
            num_exog_features=0
        )

        target = torch.randn(batch_size, 1, T_in)
        t_span = torch.linspace(0, 1, steps=T_in)

        preds, pred_logvars = model(target, t_span, exog=None)

        assert preds.shape == (batch_size, 7, 1)
        assert pred_logvars.shape == (batch_size, 7, 1)

    def test_apdtflow_all_fusion_types(self):
        """Test all fusion types with APDTFlow."""
        from apdtflow.models.apdtflow import APDTFlow

        for fusion_type in ['concat', 'gated', 'attention']:
            model = APDTFlow(
                num_scales=3,
                input_channels=1,
                filter_size=5,
                hidden_dim=16,
                output_dim=1,
                forecast_horizon=7,
                use_embedding=True,
                num_exog_features=3,
                exog_fusion_type=fusion_type
            )

            target = torch.randn(4, 1, 30)
            exog = torch.randn(4, 3, 30)
            t_span = torch.linspace(0, 1, steps=30)

            preds, pred_logvars = model(target, t_span, exog=exog)

            assert preds.shape == (4, 7, 1)
            assert pred_logvars.shape == (4, 7, 1)

    def test_apdtflow_training_with_exog(self):
        """Test training loop with exogenous variables."""
        from apdtflow.models.apdtflow import APDTFlow

        model = APDTFlow(
            num_scales=3,
            input_channels=1,
            filter_size=5,
            hidden_dim=16,
            output_dim=1,
            forecast_horizon=7,
            use_embedding=True,
            num_exog_features=3,
            exog_fusion_type='gated'
        )

        # Create dummy data
        batch_size = 8
        T_in = 30
        x = torch.randn(batch_size, 1, T_in)
        y = torch.randn(batch_size, 1, 7)
        exog = torch.randn(batch_size, 3, T_in)
        t_span = torch.linspace(0, 1, steps=T_in)

        # Training step
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        model.train()

        optimizer.zero_grad()
        preds, pred_logvars = model(x, t_span, exog=exog)
        mse = (preds - y.transpose(1, 2)) ** 2
        loss = torch.mean(0.5 * (mse / (pred_logvars.exp() + 1e-6)) + 0.5 * pred_logvars)
        loss.backward()
        optimizer.step()

        # Check that loss is finite
        assert torch.isfinite(loss)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
