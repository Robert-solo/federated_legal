import torch
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict
from transformers import LlamaConfig, LlamaForCausalLM

from fedlegal.training.dual_lora import DualLora


def test_dual_lora_adds_matrix_updates_and_freezes_shared_branch():
    torch.manual_seed(4)
    base = LlamaForCausalLM(LlamaConfig(
        vocab_size=32, hidden_size=16, intermediate_size=32,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
    ))
    config = LoraConfig(r=2, lora_alpha=2, target_modules=["q_proj"], task_type="CAUSAL_LM")
    model = get_peft_model(base, config)
    dual = DualLora(model, config)
    layer = model.base_model.model.model.layers[0].self_attn.q_proj
    with torch.no_grad():
        layer.lora_B.default.weight.fill_(0.1)
        layer.lora_B.local.weight.fill_(0.2)
    dual.activate(local=True, train="local")
    x = torch.randn(2, 16)
    actual = layer(x)
    expected = layer.base_layer(x) + x @ (
        layer.get_delta_weight("default") + layer.get_delta_weight("local")
    ).T
    assert torch.allclose(actual, expected, atol=1e-6)
    before = {k: v.clone() for k, v in get_peft_model_state_dict(model).items()}
    actual.sum().backward()
    assert layer.lora_B.local.weight.grad is not None
    assert layer.lora_B.default.weight.grad is None
    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.01)
    optimizer.step()
    assert all(torch.equal(v, get_peft_model_state_dict(model)[k]) for k, v in before.items())
    dual.reset()
    assert torch.count_nonzero(layer.lora_B.local.weight) == 0
    assert torch.count_nonzero(layer.lora_A.local.weight) > 0
