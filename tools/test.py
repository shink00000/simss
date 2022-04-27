import torch
import torch.nn.functional as F
from tqdm import tqdm

from simss.utils.config import Config


def main(args):
    cfg = Config(args.config_path, args.resume_from)

    if torch.cuda.is_available():
        device = 'cuda'
        torch.backends.cudnn.benckmark = True
    else:
        device = 'cpu'

    test_dl = cfg.build_dataloader('test')
    model = cfg.build_model().to(device)
    metric = cfg.build_metric()

    model.eval()
    with torch.no_grad():
        for image, label in tqdm(test_dl):
            image, label = image.to(device), label.to(device)
            if cfg.ms_flip:
                pred = None
                for factor in [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]:
                    image = F.interpolate(image, scale_factor=factor, mode='bilinear', align_corners=True)
                    for flip in [True, False]:
                        if flip:
                            image = image.flip(dims=[-1])
                        output = model(image)
                        tmp_pred = model.predict(output, label)
                        if flip:
                            tmp_pred = tmp_pred.flip(dims=[-1])
                        if pred is None:
                            pred = tmp_pred
                        else:
                            pred += tmp_pred
                pred /= 12
            else:
                output = model(image)
                pred = model.predict(output, label)
            metric.update(label, pred)
    metric.compute()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('config_path', type=str)
    parser.add_argument('--out_dir', type=str, default='./results')
    parser.add_argument('--resume_from', type=str, default=None)
    args = parser.parse_args()

    main(args)
