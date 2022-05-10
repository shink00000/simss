import torch
from simss.models.backbones import SwinTransformer
from collections import OrderedDict
from torchvision.transforms.functional import resize

with torch.no_grad():
    for scale in ['tiny', 'small', 'base']:

        targets = torch.load(f'./assets/original/swin_{scale}_patch4_window7_224.pth', map_location='cpu')['model']

        m = SwinTransformer(scale=scale)
        state_dict = OrderedDict()
        for name, p in m.named_parameters():
            _, block_no, block, *etc = name.split('.')
            if block == 'patch':
                module, param = etc
                if int(block_no) == 0:
                    module = {'projection': 'proj', 'norm': 'norm'}[module]
                    target_key = f'patch_embed.{module}.{param}'
                else:
                    module = {'projection': 'reduction', 'norm': 'norm'}[module]
                    target_key = f'layers.{int(block_no)-1}.downsample.{module}.{param}'
                state_dict[name] = targets[target_key]
            elif block == 'layers':
                layer_no, module, *etc = etc
                if 'norm' in module:
                    param = etc[0]
                    target_key = f'layers.{block_no}.blocks.{layer_no}.{module}.{param}'
                    state_dict[name] = targets[target_key]
                elif 'attn' == module:
                    sub, *etc = etc
                    if sub == 'attn':
                        param = etc[-1]
                        if 'in_proj' in etc[0]:
                            target_key = f'layers.{block_no}.blocks.{layer_no}.attn.qkv.{param}'
                            state_dict[name] = targets[target_key]
                        elif 'out_proj' in etc[0]:
                            target_key = f'layers.{block_no}.blocks.{layer_no}.attn.proj.{param}'
                            state_dict[name] = targets[target_key]
                        else:
                            raise NotImplementedError
                    elif sub == 'rel_pos_bias_table':
                        target_key = f'layers.{block_no}.blocks.{layer_no}.attn.relative_position_bias_table'
                        dst_table = resize(targets[target_key].view(13, 13, -1).permute(2, 0, 1), (15, 15)).view(-1)
                        state_dict[name] = dst_table
                    else:
                        raise NotImplementedError
                elif 'ffn' == module:
                    sub, param = etc
                    target_key = f'layers.{block_no}.blocks.{layer_no}.mlp.{sub}.{param}'
                    state_dict[name] = targets[target_key]
                else:
                    raise NotImplementedError
            else:
                raise NotImplementedError

        m.load_state_dict(state_dict, strict=False)
        torch.save(m.state_dict(), f'./assets/swin_{scale}.pth')
