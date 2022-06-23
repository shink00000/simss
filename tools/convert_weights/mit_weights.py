from collections import OrderedDict
import torch
from simss.models.backbones import MixTransformer


for i in range(6):

    targets = torch.load(f'./assets/original/mit_b{i}.pth', map_location='cpu')

    m = MixTransformer(f'b{i}')
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
                        param = etc[-1]
                        if etc[0].endswith('q'):
                            target_key = f'block{int(block_no)+1}.{layer_no}.attn.q.{param}'
                            target = targets[target_key]
                        else:
                            target_key = f'block{int(block_no)+1}.{layer_no}.attn.kv.{param}'
                            target = targets[target_key]
                            if etc[0].endswith('k'):
                                target = target[:target.shape[0]//2]
                            else:
                                target = target[target.shape[0]//2:]
                        state_dict[name] = target
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
