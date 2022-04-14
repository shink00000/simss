import torch
import os.path as osp
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from simss.utils.config import Config


def main(args):
    cfg = Config(args.config_path, args.resume_from)

    if torch.cuda.is_available():
        device = 'cuda'
        torch.backends.cudnn.benckmark = True
    else:
        device = 'cpu'

    train_dl = cfg.build_dataloader('train')
    val_dl = cfg.build_dataloader('val')
    model = cfg.build_model().to(device)
    optimizer = cfg.build_optimizer(model)
    scheduler = cfg.build_scheduler(optimizer)
    metric = cfg.build_metric()

    train_writer = SummaryWriter(osp.join(args.out_dir, 'train'))
    val_writer = SummaryWriter(osp.join(args.out_dir, 'val'))

    for e in range(cfg.start_epoch, cfg.epochs+1):
        model.train()
        train_loss = train_count = 0
        for image, label in tqdm(train_dl, desc=f'[{e}] train'):
            image, label = image.to(device), label.to(device)
            outputs = model(image)
            loss = model.loss(outputs, label)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            train_loss += loss * image.size(0)
            train_count += image.size(0)
        train_loss = (train_loss / train_count).item()

        model.eval()
        val_loss = val_count = 0
        evaluate = e % cfg.eval_interval == 0
        with torch.no_grad():
            for image, label in tqdm(val_dl, desc=f'[{e}] val'):
                image, label = image.to(device), label.to(device)
                outputs = model(image)
                loss = model.loss(outputs, label)
                if evaluate:
                    pred = model.predict(outputs, label)
                    metric.update(label, pred)
                val_loss += loss * image.size(0)
                val_count += image.size(0)
            val_loss = (val_loss / val_count).item()

        states = {
            'epoch': e,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict()
        }
        torch.save(states, osp.join(args.out_dir, 'latest.pth'))
        train_writer.add_scalar('Loss/train', train_loss, e)
        val_writer.add_scalar('Loss/val', val_loss, e)
        if evaluate:
            result = metric.compute()
            metric.reset()
            for metric_name, val in result.items():
                val_writer.add_scalar(f'Metric/{metric_name}', val, e)

        print(f'[{e}] loss: {train_loss:.04f}, val_loss: {val_loss:.04f}')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('config_path', type=str)
    parser.add_argument('--out_dir', type=str, default='./results/test')
    parser.add_argument('--resume_from', type=str, default=None)
    args = parser.parse_args()

    main(args)
