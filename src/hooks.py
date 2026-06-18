import torch

captured_features = []

def hook_fn(module, input, output):
    pooled = torch.nn.functional.adaptive_avg_pool2d(output, (1, 1))
    captured_features.append(pooled.view(pooled.size(0), -1))

def register_hooks(model):
    layers = [model.layer1, model.layer2, model.layer3, model.layer4]
    for l in layers:
        l.register_forward_hook(hook_fn)
    return layers