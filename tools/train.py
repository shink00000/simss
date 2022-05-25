import os.path as osp

import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

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

    writer = SummaryWriter(args.out_dir)

    for e in range(cfg.start_epoch, cfg.epochs+1):
        train_loss = train_count = val_loss = val_count = 0

        model.train()
        for images, labels in tqdm(train_dl, desc=f'[{e}] train'):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = model.loss(outputs, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            train_loss += loss * images.size(0)
            train_count += images.size(0)
        train_loss = (train_loss / train_count).item()

        model.eval()
        with torch.no_grad():
            for images, labels in tqdm(val_dl, desc=f'[{e}] val'):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = model.loss(outputs, labels)
                preds = model.predict(outputs, labels)
                metric.update(labels, preds)
                val_loss += loss * images.size(0)
                val_count += images.size(0)
            val_loss = (val_loss / val_count).item()
            val_result = metric.compute()
            metric.reset()

        writer.add_scalar('Loss/train', train_loss, e)
        writer.add_scalar('Loss/val', val_loss, e)
        for i, last_lr in enumerate(scheduler.get_last_lr()):
            writer.add_scalar(f'LearningRate/lr_{i}', last_lr, e)
        for metric_name, val in val_result.items():
            writer.add_scalar(f'Metric/{metric_name}', val, e)

        states = {
            'epoch': e,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict()
        }
        torch.save(states, osp.join(args.out_dir, 'latest.pth'))

        print(f'[{e}] loss: {train_loss:.04f}, val_loss: {val_loss:.04f}')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('config_path', type=str)
    parser.add_argument('--out_dir', type=str, default='./results')
    parser.add_argument('--resume_from', type=str, default=None)
    args = parser.parse_args()

    main(args)
