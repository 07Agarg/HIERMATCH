### Use this only when Level-3 (finest-level) labeled samples are available in HIERMATCH.

from __future__ import print_function

import argparse
import os
import shutil
import time
import random

import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torch.nn.functional as F

import wideresnet as models
import cifar100 as dataset
import wandb

from cifar100_get_tree_target_3 import *

from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig
from tensorboardX import SummaryWriter

parser = argparse.ArgumentParser(description='PyTorch MixMatch Training')
# Optimization options
parser.add_argument('--epochs', default=500, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--batch-size', default=64, type=int, metavar='N',
                    help='train batchsize')
parser.add_argument('--lr', '--learning-rate', default=0.002, type=float,
                    metavar='LR', help='initial learning rate')
# Checkpoints
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
# Miscs
parser.add_argument('--manualSeed', type=int, default=0, help='manual seed')
#Device options
parser.add_argument('--gpu', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
#Method options
parser.add_argument('--n-labeled', type=int, default=5000,
                        help='Number of labeled data')
parser.add_argument('--train-iteration', type=int, default=1024,
                        help='Number of iteration per epoch')
parser.add_argument('--out', default='result',
                        help='Directory to output the result')
parser.add_argument('--alpha', default=0.75, type=float)
parser.add_argument('--lambda-u', default=150, type=float)
parser.add_argument('--T', default=0.5, type=float)
parser.add_argument('--ema-decay', default=0.999, type=float)
parser.add_argument('--mix-precision', default=True, type=bool)
parser.add_argument('--num_levels', default=3, type=int, help='2 | 3')

args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}

# Use CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
use_cuda = torch.cuda.is_available()

# Random seed
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)

RANDOM_SEED = args.manualSeed
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

torch.cuda.manual_seed_all(RANDOM_SEED)
os.environ["PYTHONHASHSEED"] = str(RANDOM_SEED)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

best_acc = 0  # best test accuracy

config = {
    'T' : args.T,
    'ALPHA' : args.alpha,
    'LAMBDA_U' : args.lambda_u,
    'RANDOM_STATE' : args.manualSeed,
    'BATCH_SIZE' : args.batch_size,
    'Technique' : 'MixMatch',
    'partial_samples': 0,
    'hierarchy': 3,
    'Model' : 'Wideresnet-28-8',
    'Labeled samples' : 10000,
    'Validation samples' : 5000,
    'Unlabeled samples' : 35000,
    'train transforms' : 'RandomHorizontalFlip',
    'val transforms' : '-',
    'optimizer' : 'Adam',
    'lr' : args.lr,
    'Dataset' : 'CIFAR-100',
    'epochs' : args.epochs,
    'iterations_per_epoch' : args.train_iteration,
    'ema_decay' : args.ema_decay
}
wandb.init(project='mixmatch-cifar', entity='fgvc', config=config)
args.out = wandb.run.name

def main():
    global best_acc

    if not os.path.isdir(args.out):
        mkdir_p(args.out)
    
    # Data
    print(f'==> Preparing cifar100')
    transform_train = transforms.Compose([
        dataset.RandomPadandCrop(32),
        dataset.RandomFlip(),
        dataset.ToTensor(),
    ])

    transform_val = transforms.Compose([
        dataset.ToTensor(),
    ])
    
    n_labeled = 10000
    if args.num_levels == 2:
        n_classes = [100, 20]
    if args.num_levels == 3:
        n_classes = [100, 20, 8]    

    NUM_WORKERS = 8
    print("Number of workers: ", NUM_WORKERS)

    train_labeled_set, train_unlabeled_set, val_set, test_set = dataset.get_cifar100('/home/ashimag/Datasets/cifar100/', n_labeled,
     transform_train=transform_train, transform_val=transform_val)
    
    labeled_trainloader = data.DataLoader(train_labeled_set, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=NUM_WORKERS)
    unlabeled_trainloader = data.DataLoader(train_unlabeled_set, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=NUM_WORKERS)
    val_loader = data.DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=NUM_WORKERS)
    test_loader = data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=NUM_WORKERS)
    
    # Model
    print("==> creating WRN-28-8")

    def create_model(ema=False):
        model = models.WideResNet(num_classes=n_classes[0], widen_factor=8)
        if args.num_levels == 3:
            model = models.model_bn_3(model, feature_size=512, classes=n_classes)
        if args.num_levels == 2:
            model = models.model_bn_2(model, feature_size=512, classes=n_classes)
        model = model.cuda()
        if ema:
            for param in model.parameters():
                param.detach_()

        return model

    model = create_model()
    ema_model = create_model(ema=True)
    wandb.watch(model, log='all')
    wandb.watch(ema_model, log='all')

    cudnn.benchmark = True
    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))
    

    train_criterion = SemiLoss()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    ema_optimizer= WeightEMA(model, ema_model, alpha=args.ema_decay)
    start_epoch = 0

    # Resume
    title = 'noisy-cifar-100'
    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isfile(args.resume), 'Error: no checkpoint directory found!'
        args.out = os.path.dirname(args.resume)
        checkpoint = torch.load(args.resume)
        best_acc = checkpoint['best_acc']
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        ema_model.load_state_dict(checkpoint['ema_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        logger = Logger(os.path.join(args.out, 'log.txt'), title=title, resume=True)
    else:
        logger = Logger(os.path.join(args.out, 'log.txt'), title=title)
        logger.set_names(['Train Loss', 'Train Loss X', 'Train Loss U',  'Valid Loss', 'Valid Acc.', 'Test Loss', 'Test Acc.'])

    writer = SummaryWriter(args.out)
    step = 0
    test_accs = []
    # Train and val
    scaler = torch.cuda.amp.GradScaler(enabled=args.mix_precision)
    
    for epoch in range(start_epoch, args.epochs):

        print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, state['lr']))

        train_loss, train_loss_x, train_loss_u = train(labeled_trainloader, unlabeled_trainloader, model, optimizer, ema_optimizer, train_criterion, epoch, use_cuda, writer, n_classes, scaler)
        _, train_acc = validate(labeled_trainloader, ema_model, criterion, epoch, use_cuda, mode='Train Stats')
        val_loss, val_acc = validate(val_loader, ema_model, criterion, epoch, use_cuda, mode='Valid Stats')
        test_loss, test_acc = validate(test_loader, ema_model, criterion, epoch, use_cuda, mode='Test Stats ')

        step = args.train_iteration * (epoch + 1)

        writer.add_scalar('losses/train_loss', train_loss, step)
        writer.add_scalar('losses/valid_loss', val_loss, step)
        writer.add_scalar('losses/test_loss', test_loss, step)

        writer.add_scalar('accuracy/train_acc', train_acc, step)
        writer.add_scalar('accuracy/val_acc', val_acc, step)
        writer.add_scalar('accuracy/test_acc', test_acc, step)
        
        wandb.log({
            'train loss' : train_loss,
            'train acc' : train_acc,
            'val loss' : val_loss,
            'val acc' : val_acc,
            'mixmatch loss label' : train_loss_x,
            'mixmatch loss unlab' : train_loss_u,
        })
        
        # append logger file
        logger.append([train_loss, train_loss_x, train_loss_u, val_loss, val_acc, test_loss, test_acc])

        # save model
        is_best = val_acc > best_acc
        best_acc = max(val_acc, best_acc)
        save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'ema_state_dict': ema_model.state_dict(),
                'acc': val_acc,
                'best_acc': best_acc,
                'optimizer' : optimizer.state_dict(),
            }, is_best)
        test_accs.append(test_acc)
    logger.close()
    writer.close()

    print('Best acc:')
    print(best_acc)

    print('Mean acc:')
    print(np.mean(test_accs[-20:]))


def train(labeled_trainloader, unlabeled_trainloader, model, optimizer, ema_optimizer, criterion, epoch, use_cuda, writer, n_classes, scaler):
    

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    losses_x = AverageMeter()
    losses_u = AverageMeter()
    _ws = AverageMeter()
    end = time.time()

    bar = Bar('Training', max=args.train_iteration)
    
    labeled_train_iter = iter(labeled_trainloader)
    unlabeled_train_iter = iter(unlabeled_trainloader)
    
    model.train()
    
    for batch_idx in range(args.train_iteration):
        
        try:
            inputs_x, targets_x = labeled_train_iter.next()
        except:
            labeled_train_iter = iter(labeled_trainloader)
            inputs_x, targets_x = labeled_train_iter.next()

        try:
            (inputs_u, inputs_u2), _ = unlabeled_train_iter.next()
        except:
            unlabeled_train_iter = iter(unlabeled_trainloader)
            (inputs_u, inputs_u2), _ = unlabeled_train_iter.next()
        
        # measure data loading time
        data_time.update(time.time() - end)

        batch_size = inputs_x.size(0)
        
        targets_xs = get_order_family_target(targets_x) #from fine to coarse
        
        # Transform label to one-hot
        targets_xs = [torch.zeros(batch_size, n_classes[i]).scatter_(1, targets_xs[i].view(-1,1).long(), 1) for i in range(len(n_classes))] #from fine to coarse
        
        if use_cuda:
            inputs_x, targets_xs = inputs_x.cuda(), [targets_xs[i].cuda(non_blocking=True) for i in range(len(n_classes))] #from fine to coarse
            inputs_u = inputs_u.cuda()
            inputs_u2 = inputs_u2.cuda()

        with torch.cuda.amp.autocast(enabled = args.mix_precision):
            with torch.no_grad():
                # compute guessed labels of unlabel samples
                outputs_us = model(inputs_u) #from fine to coarse
                outputs_u2s = model(inputs_u2) #from fine to coarse

        ps = [(torch.softmax(outputs_us[i], dim=1) + torch.softmax(outputs_u2s[i], dim=1)) / 2 for i in range(len(n_classes))]
        pts = [ps[i]**(1/args.T) for i in range(len(ps))]
        targets_us = [pts[i] / pts[i].sum(dim=1, keepdim=True) for i in range(len(pts))]
        targets_us = [targets_us[i].detach() for i in range(len(targets_us))]
        
        # mixup
        all_inputs = torch.cat([inputs_x, inputs_u, inputs_u2], dim=0)
        all_targets = [torch.cat([targets_xs[i], targets_us[i], targets_us[i]], dim=0) for i in range(len(n_classes))]#from fine to coarse

        l = np.random.beta(args.alpha, args.alpha)
        l = max(l, 1-l)

        idx = torch.randperm(all_inputs.size(0))

        input_a, input_b = all_inputs, all_inputs[idx]
        target_as, target_bs = all_targets, [all_targets[i][idx] for i in range(len(n_classes))] #from fine to coarse

        mixed_input = l * input_a + (1 - l) * input_b
        mixed_targets = [l * target_as[i] + (1 - l) * target_bs[i] for i in range(len(n_classes))]

        # interleave labeled and unlabeled samples between batches to get correct batchnorm calculation 
        mixed_input = list(torch.split(mixed_input, batch_size))
        mixed_input = interleave(mixed_input, batch_size)

        logits = [[] for _ in range(len(n_classes))] #from fine to coarse
               
        for input in mixed_input:
            with torch.cuda.amp.autocast(enabled = args.mix_precision):
                temp_logits = model(input) #array from fine to coarse
            
            for i in range(len(n_classes)):
                logits[i].append(temp_logits[i])
        
        # put interleaved samples back
        logits = [interleave(logits[i], batch_size) for i in range(len(n_classes))]
        logits_xs = [logits[i][0] for i in range(len(n_classes))]
        logits_us = [torch.cat(logits[i][1:], dim=0) for i in range(len(n_classes))]

        with torch.cuda.amp.autocast(enabled = args.mix_precision):
            loss_details = [criterion(logits_xs[i], mixed_targets[i][:batch_size], logits_us[i], mixed_targets[i][batch_size:], epoch + batch_idx/args.train_iteration) for i in range(len(n_classes))] #from fine to coarse
            
            Lxs = [loss_details[i][0] for i in range(len(n_classes))]
            Lus = [loss_details[i][1] for i in range(len(n_classes))]
            ws = [loss_details[i][2] for i in range(len(n_classes))]
            
            loss = sum([Lxs[i] + ws[i] * Lus[i] for i in range(len(n_classes))])
            if loss.isnan().item():
                print('Loss is NaN')
                print('Printing indiv sums')
                for i in range(len(n_classes)):
                    print(f'Sum {i+1} -> {Lxs[i] + ws[i] * Lus[i]}')

        writer.add_scalar('Unlabeled_loss_weight_lambda', ws[0], epoch)

        # record loss
        losses.update(loss.item(), inputs_x.size(0))

        for i in range(len(n_classes)):
            losses_x.update(Lxs[i].item(), inputs_x.size(0))
            losses_u.update(Lus[i].item(), inputs_x.size(0))
            _ws.update(ws[i], inputs_x.size(0))

        optimizer.zero_grad()
        
        if args.mix_precision:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        
        ema_optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | Loss_x: {loss_x:.4f} | Loss_u: {loss_u:.4f} | W: {w:.4f}'.format(
                    batch=batch_idx + 1,
                    size=args.train_iteration,
                    data=data_time.avg,
                    bt=batch_time.avg,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss=losses.avg,
                    loss_x=losses_x.avg,
                    loss_u=losses_u.avg,
                    w=_ws.avg,
                    )
        
        bar.next()
    bar.finish()

    return (losses.avg, losses_x.avg, losses_u.avg,)

def validate(valloader, model, criterion, epoch, use_cuda, mode):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    bar = Bar(f'{mode}', max=len(valloader))
    with torch.no_grad():
        for batch_idx, (inputs, family_targets) in enumerate(valloader):
            # measure data loading time
            data_time.update(time.time() - end)

            if use_cuda:
                inputs, family_targets = inputs.cuda(), family_targets.cuda(non_blocking=True)
            # compute output
            family_outputs = model(inputs)[0]
            loss = criterion(family_outputs, family_targets)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(family_outputs, family_targets, topk=(1, 5))
            losses.update(loss.item(), inputs.size(0))
            top1.update(prec1.item(), inputs.size(0))
            top5.update(prec5.item(), inputs.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # plot progress
            bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                        batch=batch_idx + 1,
                        size=len(valloader),
                        data=data_time.avg,
                        bt=batch_time.avg,
                        total=bar.elapsed_td,
                        eta=bar.eta_td,
                        loss=losses.avg,
                        top1=top1.avg,
                        top5=top5.avg,
                        )
            bar.next()
        bar.finish()
    return (losses.avg, top1.avg)

def save_checkpoint(state, is_best, checkpoint=args.out, filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))

def linear_rampup(current, rampup_length=args.epochs):
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current / rampup_length, 0.0, 1.0)
        return float(current)

class SemiLoss(object):
    def __call__(self, outputs_x, targets_x, outputs_u, targets_u, epoch):
        probs_u = torch.softmax(outputs_u, dim=1)

        Lx = -torch.mean(torch.sum(F.log_softmax(outputs_x, dim=1) * targets_x, dim=1))
        Lu = torch.mean((probs_u - targets_u)**2)

        return Lx, Lu, args.lambda_u * linear_rampup(epoch)

class WeightEMA(object):
    def __init__(self, model, ema_model, alpha=0.999):
        self.model = model
        self.ema_model = ema_model
        self.alpha = alpha
        self.params = list(model.state_dict().values())
        self.ema_params = list(ema_model.state_dict().values())
        self.wd = 0.02 * args.lr

        for param, ema_param in zip(self.params, self.ema_params):
            param.data.copy_(ema_param.data)

    def step(self):
        one_minus_alpha = 1.0 - self.alpha
        for param, ema_param in zip(self.params, self.ema_params):
            if ema_param.dtype==torch.float32:
                ema_param.mul_(self.alpha)
                ema_param.add_(param * one_minus_alpha)
                # customized weight decay
                param.mul_(1 - self.wd)

def interleave_offsets(batch, nu):
    groups = [batch // (nu + 1)] * (nu + 1)
    for x in range(batch - sum(groups)):
        groups[-x - 1] += 1
    offsets = [0]
    for g in groups:
        offsets.append(offsets[-1] + g)
    assert offsets[-1] == batch
    return offsets


def interleave(xy, batch):
    nu = len(xy) - 1
    offsets = interleave_offsets(batch, nu)
    xy = [[v[offsets[p]:offsets[p + 1]] for p in range(nu + 1)] for v in xy]
    for i in range(1, nu + 1):
        xy[0][i], xy[i][i] = xy[i][i], xy[0][i]
    return [torch.cat(v, dim=0) for v in xy]

if __name__ == '__main__':
    main()
