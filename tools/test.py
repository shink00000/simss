import torch
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
            output = model(image)
            pred = model.predict(output, label)
            metric.update(label, pred)
    metric.compute()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('config_path', type=str)
    parser.add_argument('--resume_from', type=str, default=None)
    args = parser.parse_args()

    main(args)
