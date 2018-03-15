import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import numpy as np
from tqdm import tqdm
from glob import glob
import os
from flownet import FlowNetS
import shutil
import time


class ModelAndLoss(nn.Module):
    def __init__(self, model, loss):
        super(ModelAndLoss, self).__init__()
        self.model = model
        self.loss = loss

    def forward(self, data, target, inference=False):
        output = self.model(data)
        loss_values = self.loss(output, target)
        if not inference:
            return loss_values
        else:
            return loss_values, output


class IteratorTimer():
    def __init__(self, iterable):
        self.iterable = iterable
        self.iterator = self.iterable.__iter__()

    def __iter__(self):
        return self

    def __len__(self):
        return len(self.iterable)

    def __next__(self):
        start = time.time()
        n = self.iterator.next()
        self.last_duration = (time.time() - start)
        return n


def train(model, optimizer, epoch, start_iteration, data_loader, n_batches, logger, log_freq, scheduler=None, is_validate=False, offset=0):


    statistics = []
    total_loss = 0

    if is_validate:
        model.eval()
        title = 'Validating Epoch {}'.format(epoch)
        progress = tqdm(IteratorTimer(data_loader), ncols=100,
                        total=np.minimum(len(data_loader), n_batches), leave=True, position=offset,
                        desc=title)
    else:
        model.train()
        title = 'Training Epoch {}'.format(epoch)
        progress = tqdm(IteratorTimer(data_loader), ncols=120,
                        total=np.minimum(len(data_loader), n_batches), smoothing=.9, miniters=1, leave=True,
                        position=offset, desc=title)
    last_log_time = progress._time()

    for batch_id, (data, target) in enumerate(progress):
        data, target = [Variable(d, volatile=is_validate) for d in data], [Variable(t, volatile=is_validate) for t in
                                                                           target]
        if torch.cuda.is_available():
            data, target = [d.cuda(async=True) for d in data], [t.cuda(async=True) for t in target]

        optimizer.zero_grad() if not is_validate else None
        losses = model(data[0], target[0])
        losses = [torch.mean(loss_value) for loss_value in losses]
        loss_val = losses[0]  # Collect first loss for weight update
        total_loss += loss_val.data[0]
        loss_values = [v.data[0] for v in losses]

        loss_labels = list(model.module.loss.loss_labels)

        if not is_validate:
            loss_val.backward()
            optimizer.step()
        global_iteration = start_iteration + batch_id

        if not is_validate:
            scheduler.step() if scheduler is not None else None
            loss_labels.append('lr')
            loss_values.append(optimizer.param_groups[0]['lr'])

        loss_labels.append('load')
        loss_values.append(progress.iterable.last_duration)
        statistics.append(loss_values)
        title = '{} Epoch {}'.format('Validating' if is_validate else 'Training', epoch)

        progress.set_description(title + ' ' + str({label: statistic for label, statistic in zip(loss_labels, statistics[-1])}))

        if ((((global_iteration + 1) % log_freq) == 0 and not is_validate) or
                (is_validate and batch_id == n_batches - 1)):

            global_iteration = global_iteration if not is_validate else start_iteration

            logger.add_scalar('batch logs per second', len(statistics) / (progress._time() - last_log_time),
                              global_iteration)
            last_log_time = progress._time()

            all_losses = np.array(statistics)

            for i, key in enumerate(loss_labels):
                logger.add_scalar('average batch ' + str(key), all_losses[:, i].mean(), global_iteration)
                logger.add_histogram(str(key), all_losses[:, i], global_iteration)

        # Reset Summary
        statistics = []

        if (batch_id == n_batches):
            break



    progress.close()

    return total_loss / float(batch_id + 1), (batch_id + 1)

def save_checkpoint(state, is_best, path, prefix, filename='checkpoint.pth.tar'):
    prefix_save = os.path.join(path, prefix)
    name = prefix_save + '_' + filename
    torch.save(state, name)
    if is_best:
        shutil.copyfile(name, prefix_save + '_model_best.pth.tar')

if __name__ == "__main__":

    start_epoch = 0
    total_epochs = 1000
    skip_validation = True
    validation_frequency = 10
    model = FlowNetS(in_channels=6)
    loss = nn.L1Loss()
    model_and_loss = ModelAndLoss(model, loss)

    init_lr = 5e-5

    optimizer = torch.optim.Adam(init_lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=3e3, eta_min=1e-6)

    save = "./logs"
    if not os.path.exists(save):
        os.makedirs(save)
    train_logger = SummaryWriter(log_dir=os.path.join(save, 'train'), comment='training')
    validation_logger = SummaryWriter(log_dir=os.path.join(save, 'validation'), comment='validation')

    best_err = 1e8
    progress = tqdm(range(start_epoch, total_epochs + 1), miniters=1, ncols=100, desc='Overall Progress',
                    leave=True, position=0)
    offset = 1
    last_epoch_time = progress._time()
    global_iteration = 0


    for epoch in progress:
        if not skip_validation and ((epoch - 1) % validation_frequency) == 0:
            validation_loss, _ = train(epoch=epoch - 1, start_iteration=global_iteration,
                                       data_loader=validation_loader, model=model_and_loss, n_batches=200, optimizer=optimizer,
                                       logger=validation_logger, is_validate=True, offset=offset)
            offset += 1

            is_best = False
            if validation_loss < best_err:
                best_err = validation_loss
                is_best = True

            checkpoint_progress = tqdm(ncols=100, desc='Saving Checkpoint', position=offset)
            save_checkpoint({'arch': model,
                                   'epoch': epoch,
                                   'state_dict': model_and_loss.module.model.state_dict(),
                                   'best_EPE': best_err},
                                  is_best, save, model)
            checkpoint_progress.update(1)
            checkpoint_progress.close()
            offset += 1

        train_loss, iterations = train(epoch=epoch, start_iteration=global_iteration,
                                       data_loader=train_loader, model=model_and_loss, n_batches=len(train_loader), optimizer=optimizer,
                                       logger=train_logger, log_freq=10, offset=offset)
        global_iteration += iterations
        offset += 1

        # save checkpoint after every validation_frequency number of epochs
        if ((epoch - 1) % validation_frequency) == 0:
            checkpoint_progress = tqdm(ncols=100, desc='Saving Checkpoint', position=offset)
            save_checkpoint({'arch': model,
                                   'epoch': epoch,
                                   'state_dict': model_and_loss.module.model.state_dict(),
                                   'best_EPE': train_loss},
                                  False, save, model, filename='train-checkpoint.pth.tar')
            checkpoint_progress.update(1)
            checkpoint_progress.close()
