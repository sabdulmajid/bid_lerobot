import torch
from torch import nn

from lerobot.common.policies.vqbet.configuration_vqbet import VQBeTConfig
from lerobot.common.policies.vqbet.modeling_vqbet import VQBeTHead, VQBeTModel


def _small_config() -> VQBeTConfig:
    return VQBeTConfig(
        n_obs_steps=2,
        n_action_pred_token=2,
        action_chunk_size=2,
        input_shapes={
            "observation.image": [3, 96, 96],
            "observation.state": [2],
        },
        output_shapes={"action": [2]},
        crop_shape=(84, 84),
        vqvae_n_embed=5,
        vqvae_embedding_dim=8,
        vqvae_enc_hidden_dim=8,
        gpt_input_dim=8,
        gpt_output_dim=8,
        gpt_n_layer=1,
        gpt_n_head=1,
        gpt_hidden_dim=8,
        mlp_hidden_dim=8,
        sequentially_select=False,
    )


def test_vqbet_head_code_log_prob_matches_manual_gather():
    cfg = _small_config()
    head = VQBeTHead(cfg)
    batch_size = 3
    n_tokens = 2
    n_layers = head.vqvae_model.vqvae_num_layers
    x = torch.randn(batch_size, n_tokens, cfg.gpt_output_dim)
    code_ids = torch.tensor(
        [[0, 1], [2, 3], [4, 0], [1, 2], [3, 4], [0, 0]],
        dtype=torch.long,
    )
    temperature = 0.7

    out = head(x, temperature=temperature, sampled_centers=code_ids)
    manual = torch.log_softmax(out["cbet_logits"] / temperature, dim=-1)
    manual = manual.gather(-1, code_ids.unsqueeze(-1)).squeeze(-1).sum(dim=-1)

    assert out["sampled_centers"].shape == (batch_size * n_tokens, n_layers)
    assert out["sampled_log_prob"].shape == (batch_size * n_tokens,)
    assert out["code_entropy"].shape == (batch_size * n_tokens,)
    assert torch.allclose(out["sampled_log_prob"], manual)
    assert torch.isfinite(out["code_entropy"]).all()


def test_vqbet_head_recompute_uses_supplied_code_ids_without_resampling():
    cfg = _small_config()
    head = VQBeTHead(cfg)
    x = torch.randn(2, 3, cfg.gpt_output_dim)
    fixed_ids = torch.randint(0, cfg.vqvae_n_embed, (6, head.vqvae_model.vqvae_num_layers))

    torch.manual_seed(1)
    first = head(x, temperature=0.5, sampled_centers=fixed_ids)
    torch.manual_seed(999)
    second = head(x, temperature=0.5, sampled_centers=fixed_ids)

    assert torch.equal(first["sampled_centers"], fixed_ids)
    assert torch.equal(second["sampled_centers"], fixed_ids)
    assert torch.allclose(first["sampled_log_prob"], second["sampled_log_prob"])


class _TinyRgbEncoder(nn.Module):
    def __init__(self, out_dim: int):
        super().__init__()
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.zeros(x.shape[0], self.out_dim, device=x.device, dtype=x.dtype)


def _tiny_model() -> VQBeTModel:
    cfg = _small_config()
    model = VQBeTModel(cfg)
    model.rgb_encoder = _TinyRgbEncoder(cfg.gpt_input_dim)
    model.rgb_feature_projector = nn.Identity()
    model.state_projector = nn.Linear(cfg.input_shapes["observation.state"][0], cfg.gpt_input_dim)
    model.policy = nn.Identity()
    return model


def _temporal_batch(batch_size: int = 2) -> dict[str, torch.Tensor]:
    cfg = _small_config()
    return {
        "observation.state": torch.randn(batch_size, cfg.n_obs_steps, cfg.input_shapes["observation.state"][0]),
        "observation.images": torch.randn(batch_size, cfg.n_obs_steps, 1, *cfg.input_shapes["observation.image"]),
    }


def test_vqbet_model_code_policy_returns_current_token_policy_stats_shapes():
    model = _tiny_model()
    batch = _temporal_batch(batch_size=2)

    out = model.code_policy(batch, temperature=0.5)

    assert out["action_pred"].shape == (2, model.config.action_chunk_size, model.config.output_shapes["action"][0])
    assert out["code_ids"].shape == (2, model.action_head.vqvae_model.vqvae_num_layers)
    assert out["code_logits"].shape == (2, model.action_head.vqvae_model.vqvae_num_layers, model.config.vqvae_n_embed)
    assert out["log_prob"].shape == (2,)
    assert out["entropy"].shape == (2,)
    assert out["value"].shape == (2,)
    assert torch.isfinite(out["log_prob"]).all()
    assert torch.isfinite(out["entropy"]).all()
    assert torch.isfinite(out["value"]).all()


def test_vqbet_code_policy_recomputed_old_log_prob_and_ratio_one_before_update():
    model = _tiny_model()
    model.eval()
    batch = _temporal_batch(batch_size=3)

    torch.manual_seed(123)
    rollout = model.code_policy(batch, temperature=0.5)
    recomputed = model.code_policy(batch, code_ids=rollout["code_ids"], temperature=0.5)

    assert torch.allclose(recomputed["log_prob"], rollout["log_prob"])
    ratio = torch.exp(recomputed["log_prob"] - rollout["log_prob"])
    assert torch.allclose(ratio, torch.ones_like(ratio))


def test_vqbet_value_head_overfits_tiny_feature_batch():
    cfg = _small_config()
    value_head = nn.Linear(cfg.gpt_output_dim, 1)
    optimizer = torch.optim.Adam(value_head.parameters(), lr=5e-2)
    features = torch.randn(16, cfg.gpt_output_dim)
    target = torch.linspace(-1.0, 1.0, 16)

    initial_loss = torch.nn.functional.mse_loss(value_head(features).squeeze(-1), target)
    for _ in range(300):
        loss = torch.nn.functional.mse_loss(value_head(features).squeeze(-1), target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    final_loss = torch.nn.functional.mse_loss(value_head(features).squeeze(-1), target)
    assert final_loss < initial_loss * 0.05
