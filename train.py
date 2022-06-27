import argparse
import json
import torch
import os
import time
from models import ResNet34
from env import AttrDict
from dataset import AccentDataset
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from utils import scan_checkpoint, load_checkpoint, save_checkpoint


def train(args, hparams):
    torch.cuda.manual_seed(hparams.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    steps = 0
    model = ResNet34(hparams.blocks_each_layer, num_classes=hparams.num_classes)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), hparams.learning_rate, betas=[hparams.adam_b1, hparams.adam_b2])

    if os.path.isdir(args.checkpoint_path):
        checkpoint = scan_checkpoint(a.checkpoint_path, 'checkpoint_')

    if checkpoint is None:
        state_dict = None
        last_epoch = -1
    else:
        state_dict = load_checkpoint(checkpoint, device)
        model.load_state_dict(state_dict['model'])
        steps = state_dict['steps'] + 1
        last_epoch = state_dict['epoch']
        optimizer.load_state_dict(state_dict['optim'])

    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=hparams.lr_decay, last_epoch=last_epoch)
    
    trainset = AccentDataset(args.training_file, hparams)
    train_loader = DataLoader(trainset, num_workers=hparams.num_workers, shuffle=True, batch_size=hparams.batch_size)
    
    valset = AccentDataset(args.validation_file, hparams)
    val_loader = DataLoader(valset, num_workers=1, shuffle=False, batch_size=1)
    # tensorboard
    sw = SummaryWriter(os.path.join(args.checkpoint_path, 'logs'))

    loss = CrossEntropyLoss(reduction='mean')
    model.train()
    # iterate over epochs
    for epoch in range(max(0, last_epoch), args.training_epochs):
        start = time.time()
        print("Epoch: {}".format(epoch+1))
        # iterate over batches
        for i, batch in enumerate(train_loader):
            start_b = time.time()
            x, y = batch

            x = torch.autograd.Variable(x.to(device))
            y = torch.autograd.Variable(y.to(device))

            x = torch.unsqueeze(x, 1)

            print(f"x shape: {x.size()}, y shape: {y.size()}")

            y_hat = model(x)

            optimizer.zero_grad()
            loss_train = loss(y_hat, y)
            loss_train.backward()
            optimizer.step()

            if steps % args.log_interval == 0:
                with torch.no_grad():
                    print(f"Steps: {steps}, Loss Total: {loss_train}, time spent: {time.time()-start_b}")

            # checkpointing
            if steps % args.checkpoint_interval == 0 and steps != 0:
                checkpoint_path = f"{args.checkpoint_path}/checkpoint_{steps}"
                save_checkpoint(checkpoint_path,{'model': model.state_dict(), 
                                                'optim': optimizer.state_dict(),
                                                'epoch': epoch, 
                                                'steps': steps})
            
            if steps % args.summary_interval == 0:
                sw.add_scalar("loss/train", loss_train, steps)
            
            # Validation
            if steps % args.validation_interval == 0:
                model.eval()
                torch.cuda.empty_cache()
                val_err_total = 0
                with torch.no_grad():
                    for j, batch in enumerate(val_loader):
                        x, y = batch
                        x = torch.unsqueeze(x, 1)
                        y_hat = model(x.to(device))

                        loss_val = loss(y_hat, y)
                        val_err_total += loss_val
                    
                    val_err_avg = val_err_total / (j+1)
                    sw.add_scalar('loss/eval', val_err_avg, steps)
            
                model.train()
            
            steps += 1
            scheduler.step()

        print(f"Time taken for epoch {epoch+1} is {int(time.time()-start)} sec \n")
            

if __name__ == '__main__':
    print('Initializing Training Process...')

    parser = argparse.ArgumentParser()

    parser.add_argument('--checkpoint_path', default='checkpoint')
    parser.add_argument('--training_file', default='data/training.txt')
    parser.add_argument('--validation_file', default='data/validation.txt')
    parser.add_argument('--config', default='config/config_ac.json')
    parser.add_argument('--training_epochs', default=3100, type=int)
    parser.add_argument('--stdout_interval', default=5, type=int)
    parser.add_argument('--checkpoint_interval', default=5000, type=int)
    parser.add_argument('--summary_interval', default=100, type=int)
    parser.add_argument('--log_interval', default=10, type=int)
    parser.add_argument('--validation_interval', default=1000, type=int)

    a = parser.parse_args()

    with open(a.config) as f:
        data = f.read()
    
    json_config = json.loads(data)
    h = AttrDict(json_config)

    torch.manual_seed(h.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(h.seed)
        h.num_gpus = torch.cuda.device_count()
        h.batch_size = int(h.batch_size / h.num_gpus)
        print('Batch size per GPU :', h.batch_size)
    else:
        pass

    if h.num_gpus > 1:
        # multi-processing training
        mp.spawn(train, nprocs=h.num_gpus, args=(a, h,))
    else:
        train(a, h)


                




