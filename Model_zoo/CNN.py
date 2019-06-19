import torch
import torchvision.transforms as T
from FDDB_dataloader import FDDB
from torch.utils.data import DataLoader
import tqdm
import torchvision.models as models
import torch.nn as nn
import torch.optim
import torch.optim.lr_scheduler as lr_scheduler
import time
import os
import shutil

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        
def accuracy_VOC2012(output, target):
    with torch.no_grad():
        batch_size = target.size(0)
        accur = output.gt(0.).long().eq(target.long()).float().mean()
        res = []
        res.append(accur)
        return res

class CNN():
    def __init__(self, params = {}):
        self.batch_size = self.set_params(256, 'batch_size', params)
        self.workers = self.set_params(4, 'workers', params)
        self.gpu = self.set_params(0, 'gpu', params)
        self.arch = self.set_params('resnet18', 'arch', params)
        self.optim = self.set_params('Adam', 'optim', params)
        self.lr = self.set_params(0.1, 'lr', params)
        self.step_size = self.set_params(30, 'step_size', params)
        self.gamma = self.set_params(0.1, 'gamma', params)
        self.weight_decay = self.set_params(1e-4, 'weight_decay', params)
        self.momentum = self.set_params(0.9, 'momentum', params)
        self.epochs = self.set_params(90, 'epochs', params)
        self.save_epoch = self.set_params(30, 'save_epoch', params)
        self.save = self.set_params(False, 'save', params)
        self.suffix = self.set_params('', 'suffix', params)
        self.print_freq = self.set_params(10, 'print_freq', params)
        
        self.cnn = models.__dict__[self.arch](num_classes=1)
        self.criterion = nn.BCEWithLogitsLoss()
        
        if self.gpu is not None:
            self.cnn.cuda(self.gpu)
            self.criterion.cuda(self.gpu)        
        
        if self.optim == 'SGD':
            self.optimizer = torch.optim.SGD(self.cnn.parameters(), self.lr, 
                                             momentum = self.momentum,
                                             weight_decay=self.weight_decay)
        elif self.optim == 'Adam':
            self.optimizer = torch.optim.Adam(self.cnn.parameters(), self.lr, 
                                              weight_decay = self.weight_decay)
        
        normalize = T.Normalize(mean = [0.5116, 0.4200, 0.3651],
                         std = [0.2655, 0.2430, 0.2420])
        self.img_trans = T.Compose([
                      T.Resize((96,96)),
                      T.ToTensor(),
                      normalize
                  ])
        
    def set_params(self, default, label, params):
        return default if label not in params.keys() else params[label]
    
    def train(self, epoch):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        top0 = AverageMeter()

        # switch to train mode
        self.cnn.train()
        
#         enum = tqdm.tqdm(enumerate(self.train_loader),
#                          total= len(self.train_loader),
#                         desc='Train: Loss {loss.val:.4f} ({loss.avg:.4f}) '
#                              'Acc@0 {top0.val:.3f} ({top0.avg:.3f})'
#                          .format(loss=losses, top0=top0))
        
        for i, (input, target) in enumerate(self.train_loader):

            if self.gpu is not None:
                input = input.cuda(self.gpu, non_blocking=True)
                target = target.float().unsqueeze(-1).cuda(self.gpu, non_blocking=True)

            # compute output
            output = self.cnn(input)
            loss = self.criterion(output, target)

            acc = accuracy_VOC2012(output, target)

            losses.update(loss.item(), input.size(0))
            top0.update(acc[0], input.size(0))

            # compute gradient and do SGD step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
#             if (i+1) % self.print_freq == 0:
#                 enum.set_description('Train: Loss {loss.val:.4f} ({loss.avg:.4f})\t'
#                                      'Acc@0 {top0.val:.3f} ({top0.avg:.3f})'
#                                      .format(loss=losses, top0=top0))
        return losses.avg, top0.avg

    def validate(self):
        batch_time = AverageMeter()
        losses = AverageMeter()
        top0 = AverageMeter()

        # switch to evaluate mode
        self.cnn.eval()

        with torch.no_grad():
#             enum = tqdm.tqdm(enumerate(self.val_loader), 
#                       total=len(self.val_loader), 
#                       desc='Test: Loss {loss.val:.4f} ({loss.avg:.4f}) '
#                           'Acc@0 {top0.val:.3f} ({top0.avg:.3f})'
#                       .format( loss=losses, top0=top0))
            for i, (input, target) in enumerate(self.val_loader):
                if self.gpu is not None:
                    input = input.cuda(self.gpu, non_blocking=True)
                    target = target.float().unsqueeze(-1).cuda(self.gpu, non_blocking=True)

                # compute output
                output = self.cnn(input)
                loss = self.criterion(output, target)

                acc = accuracy_VOC2012(output, target)
                losses.update(loss.item(), input.size(0))
                top0.update(acc[0], input.size(0))
#                 if (i+1) % self.print_freq == 0:
#                     enum.set_description('Test: Loss {loss.val:.4f} ({loss.avg:.4f}) '
#                                          'Acc@0 {top0.val:.3f} ({top0.avg:.3f})'
#                                     .format(loss=losses, top0=top0))
        return losses.avg, top0.avg

    def save_checkpoint(self, state, is_best, filename='checkpoint.pth.tar'):
        torch.save(state, filename)
        if is_best:
            shutil.copyfile(filename, 'model_best_{}_{}.pth.tar'
                            .format(self.arch, self.suffix))
    
    def load_checkpoint(self, resume):
        if os.path.isfile(resume):
            print("=> loading checkpoint '{}'".format(resume))
            checkpoint = torch.load(resume, map_location=torch.device("cuda:{}".format(self.gpu)))
            self.cnn.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {}) as Teacher!"
                  .format(resume, checkpoint['epoch']))
            del checkpoint
        else:
            print("=> no checkpoint found at '{}'".format(resume))
            return
    
    def fit(self):
        self.logger = {'train_acc': [],'test_acc':[],'train_loss':[],'test_loss':[]}
        best_acc1 = 0
        acc1 = 0
        acc1_test = 0
        scheduler = lr_scheduler.StepLR(self.optimizer, 
                                        step_size=self.step_size, 
                                        gamma=self.gamma)
        epochs = tqdm.tqdm(range(self.epochs), 
                               desc='Train Acc {:.4f} Test Acc {:.4f} ({:.4f}) Epoch'.format(acc1_test, acc1, best_acc1))
        for epoch in epochs:
            # train for one epoch
            loss_train, acc1_train = self.train(epoch)

            # evaluate on validation set
            loss_test, acc1 = self.validate()
            
            self.logger['train_acc'].append(acc1_train)
            self.logger['test_acc'].append(acc1)
            self.logger['train_loss'].append(loss_train)
            self.logger['test_loss'].append(loss_test)
            # remember best acc@1 and save checkpoint
            is_best = acc1 > best_acc1
            best_acc1 = max(acc1, best_acc1)
            if self.save:
                save_dir = 'checkpoint_{}_{}.pth.tar'.format(self.arch, self.suffix)
                if (epoch+1)%self.save_epoch == 0:
                    self.save_checkpoint({
                        'epoch': epoch + 1,
                        'arch': self.arch,
                        'state_dict': self.cnn.state_dict(),
                        'best_acc1': best_acc1,
                        'acc1': acc1,
                        'optimizer' : self.optimizer.state_dict(),
                    }, is_best, save_dir)
                elif is_best:
                    self.save_checkpoint({
                        'epoch': epoch + 1,
                        'arch': self.arch,
                        'state_dict': self.cnn.state_dict(),
                        'best_acc1': best_acc1,
                        'acc1': acc1,
                        'optimizer' : self.optimizer.state_dict(),
                    }, is_best, save_dir)
            epochs.set_description_str('Train Acc {:.4f} Test Acc {:.4f} ({:.4f})'
                                       .format(acc1_train, best_acc1, acc1))
            scheduler.step()
            
    def set_dataset(self, db_train, db_val):
        self.train_loader = DataLoader(db_train, batch_size=self.batch_size, shuffle=True,
                                 num_workers=self.workers, pin_memory=True)
        self.val_loader = DataLoader(db_val, batch_size=self.batch_size, shuffle=False,
                               num_workers=self.workers, pin_memory=True)
    
    def get_score(self, input):
        self.cnn.eval()

        with torch.no_grad():
            if self.gpu is not None:
                input = input.cuda(self.gpu, non_blocking=True)
#                 target = target.float().unsqueeze(-1).cuda(self.gpu, non_blocking=True)
            output = self.cnn(input)

        return output.cpu()[0]