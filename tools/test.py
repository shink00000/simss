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
            aug_pred = None
            for factor in [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]:
                image = F.interpolate(image, scale_factor=factor, mode='bilinear', align_corners=True)
                for flip in [True, False]:
                    if flip:
                        image = image.flip(dims=[-1])
                    output = model(image)
                    pred = model.predict(output, label)
                    if flip:
                        pred = pred.flip(dims=[-1])
                    if aug_pred is None:
                        aug_pred = pred
                    else:
                        aug_pred += pred
            aug_pred /= 12
            metric.update(label, aug_pred)
    metric.compute()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('config_path', type=str)
    parser.add_argument('--out_dir', type=str, default='./results')
    parser.add_argument('--resume_from', type=str, default=None)
    args = parser.parse_args()

    main(args)
