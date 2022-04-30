import torch
from simss.models.backbones import MiT
from collections import OrderedDict
import os


os.makedirs('./assets', exist_ok=True)
for i in range(6):

    targets = torch.load(f'./assets/original/mit_b{i}.pth')

    m = MiT(f'b{i}')
    state_dict = OrderedDict()
    for name, p in m.named_parameters():
        _, block_no, block, *etc = name.split('.')
        if block == 'patch':
            module, param = etc
            module = {'projection': 'proj', 'norm': 'norm'}[module]
            target_key = f'patch_embed{int(block_no)+1}.{module}.{param}'
            state_dict[name] = targets[target_key]
        elif block == 'layers':
            layer_no, module, *etc = etc
            if 'norm' in module:
                param = etc[0]
                target_key = f'block{int(block_no)+1}.{layer_no}.{module}.{param}'
                state_dict[name] = targets[target_key]
            elif 'attn' == module:
                sub, *etc = etc
                if sub == 'attn':
                    if 'in_proj' in etc[0]:
                        param = etc[0].split('_')[-1]
                        target_key_1 = f'block{int(block_no)+1}.{layer_no}.attn.q.{param}'
                        target_key_2 = f'block{int(block_no)+1}.{layer_no}.attn.kv.{param}'
                        state_dict[name] = torch.cat([targets[target_key_1], targets[target_key_2]], dim=0)
                    elif 'out_proj' in etc[0]:
                        param = etc[-1]
                        target_key = f'block{int(block_no)+1}.{layer_no}.attn.proj.{param}'
                        state_dict[name] = targets[target_key]
                    else:
                        raise NotImplementedError
                elif sub == 'reduction':
                    param = etc[0]
                    target_key = f'block{int(block_no)+1}.{layer_no}.attn.sr.{param}'
                    state_dict[name] = targets[target_key]
                elif sub == 'norm':
                    param = etc[0]
                    target_key = f'block{int(block_no)+1}.{layer_no}.attn.norm.{param}'
                    state_dict[name] = targets[target_key]
                else:
                    raise NotImplementedError
            elif 'ffn' == module:
                sub, param = etc
                sub = {'fc1': 'fc1', 'fc2': 'fc2', 'conv': 'dwconv.dwconv'}[sub]
                target_key = f'block{int(block_no)+1}.{layer_no}.mlp.{sub}.{param}'
                if sub.startswith('fc') and param == 'weight':
                    targets[target_key] = targets[target_key][..., None, None]
                state_dict[name] = targets[target_key]
            else:
                raise NotImplementedError
        elif block == 'norm':
            param = etc[0]
            target_key = f'norm{int(block_no)+1}.{param}'
            state_dict[name] = targets[target_key]
        else:
            raise NotImplementedError

    m.load_state_dict(state_dict)
    torch.save(m.state_dict(), f'./assets/mit_b{i}.pth')
