import torch
from simss.models.backbones import HRNet
from collections import OrderedDict

for width in [18, 32, 48]:

    targets = torch.load(f'./assets/original/hrnetv2_w{width}_imagenet_pretrained.pth', map_location='cpu')

    m = HRNet(width=width)
    state_dict = OrderedDict()
    for name, p in m.named_parameters():
        if name.startswith('conv'):
            num = name[4]
            if name.split('.')[1] == 'conv':
                target_key = f'conv{num}.weight'
            else:
                param = name.split('.')[2]
                target_key = f'bn{num}.{param}'
        elif name.startswith('stage1'):
            target_key = f'layer1.{name[20:]}'
        elif name.startswith('stage'):
            if name in targets:
                target_key = name
            else:
                splits = name.split('.')
                n1 = int(splits[-2]) // 3
                n2 = int(splits[-2]) % 3
                target_key = '.'.join(splits[:-2] + [str(n1), str(n2)] + [splits[-1]])
        else:
            _, _, module, param = name.split('.')
            if name.startswith('trans1.0'):
                target_key = f'transition{name[5:8]}.{0 if module == "conv" else 1}.{param}'
            else:
                target_key = f'transition{name[5:8]}.0.{0 if module == "conv" else 1}.{param}'

        state_dict[name] = targets.pop(target_key)
        if target_key.endswith('bias'):
            for k in ['running_mean', 'running_var', 'num_batches_tracked']:
                state_dict[name.replace('bias', k)] = targets.pop(target_key.replace('bias', k))

    m.load_state_dict(state_dict, strict=False)
    torch.save(m.state_dict(), f'./assets/hrnetv2_w{width}.pth')
