"""Exercise original model definitions with synthetic tensors on CPU."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F
from models import RES_Block, UNet, WC_Block


def main():
    torch.manual_seed(7)
    torch.set_num_threads(2)
    model = UNet(1, 4)
    image = torch.randn(1, 1, 32, 32)
    probabilities = model(image)
    assert probabilities.shape == (1, 4, 32, 32)
    assert torch.isfinite(probabilities).all()
    assert torch.allclose(probabilities.sum(dim=1), torch.ones(1, 32, 32), atol=1e-5)
    target = torch.randint(0, 4, (1, 32, 32))
    loss = F.nll_loss(probabilities.clamp_min(1e-8).log(), target)
    loss.backward()
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in model.parameters())
    print('PASS: original U-Net, synthetic 1×1×32×32 input, four-class output and finite gradients')

    with torch.no_grad():
        residual = RES_Block(4, 4).eval()(torch.randn(1, 4, 32, 32))
        context = WC_Block(4, 8).eval()(torch.randn(1, 4, 32, 32))
    assert residual.shape == (1, 4, 34, 34)
    assert context.shape == (1, 8, 18, 18)
    assert torch.isfinite(residual).all() and torch.isfinite(context).all()
    print('PASS: original RES block 32→34, WC block 32→18; historical geometry is preserved')
    print('No MRI data, learned weights, training run or clinical evaluation was used.')


if __name__ == '__main__':
    main()
