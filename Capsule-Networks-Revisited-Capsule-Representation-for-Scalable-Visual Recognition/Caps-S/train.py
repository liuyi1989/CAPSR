from __future__ import print_function
import argparse
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

from model import capsules
from loss import SpreadLoss

# os.environ['CUDA_VISIBLE_DEVICES'] = "6"   这是V100 的那块显卡
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
def text_create(name, msg):
    desktop_path = "./snapshots/"  
    full_path = desktop_path + name + '.txt'  
    file = open(full_path, 'a+')
    file.write(msg)   
    file.close()

# Training settings
parser = argparse.ArgumentParser(description='PyTorch Matrix-Capsules-EM')
parser.add_argument('--batch-size', type=int, default=80, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--test-batch-size', type=int, default=80, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--test-intvl', type=int, default=5, metavar='N',
                    help='test intvl (default: 1)')
parser.add_argument('--epochs', type=int, default=150, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='learning rate (default: 0.001)')
parser.add_argument('--weight-decay', type=float, default=0, metavar='WD',
                    help='weight decay (default: 0)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--em-iters', type=int, default=2, metavar='N',
                    help='iterations of EM Routing')
parser.add_argument('--snapshot-folder', type=str, default='./snapshots', metavar='SF',
                    help='where to store the snapshots')
parser.add_argument('--data-folder', type=str, default='./data', metavar='DF',
                    help='where to store the datasets')
parser.add_argument('--dataset', type=str, default='CIFAR10', metavar='D',
                    help='dataset for training(mnist, smallNORB,ImageNet)')


def get_setting(args):
    kwargs = {'num_workers': 2, 'pin_memory': True} if args.cuda else {}
    path = os.path.join(args.data_folder, args.dataset)
    if args.dataset == 'mnist':
        num_class = 10
        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST(path, train=True, download=True,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ])),
            batch_size=args.batch_size, shuffle=True, **kwargs)
        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST(path, train=False,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ])),
            batch_size=args.test_batch_size, shuffle=True, **kwargs)
    elif args.dataset == 'smallNORB':
        num_class = 5
        train_loader = torch.utils.data.DataLoader(
            smallNORB(path, train=True, download=True,
                      transform=transforms.Compose([
                          transforms.Resize(48),
                          transforms.RandomCrop(32),
                          transforms.ColorJitter(brightness=32./255, contrast=0.5),
                          transforms.ToTensor()
                      ])),
            batch_size=args.batch_size, shuffle=True, **kwargs)
        test_loader = torch.utils.data.DataLoader(
            smallNORB(path, train=False,
                      transform=transforms.Compose([
                          transforms.Resize(48),
                          transforms.CenterCrop(32),
                          transforms.ToTensor()
                      ])),
            batch_size=args.test_batch_size, shuffle=True, **kwargs)
    elif args.dataset == 'FashionMNIST':
        num_class = 10
        train_loader = torch.utils.data.DataLoader(
            datasets.FashionMNIST(path, train=True, download=True,
                      transform=transforms.Compose([
                          transforms.Resize(48),
                          transforms.RandomCrop(32),
                          transforms.ColorJitter(brightness=32./255, contrast=0.5),
                          transforms.ToTensor()
                      ])),
            batch_size=args.batch_size, shuffle=True, **kwargs)
        test_loader = torch.utils.data.DataLoader(
            datasets.FashionMNIST(path, train=False,
                      transform=transforms.Compose([
                          transforms.Resize(48),
                          transforms.CenterCrop(32),
                          transforms.ToTensor()
                      ])),
            batch_size=args.test_batch_size, shuffle=True, **kwargs)  
    elif args.dataset == 'SVHN':
        num_class = 10
        train_loader = torch.utils.data.DataLoader(
            datasets.SVHN(path, train=True, download=True,
                      transform=transforms.Compose([
                          transforms.RandomCrop(32, padding=4),
                          transforms.RandomHorizontalFlip(0.5),                          
                          transforms.ToTensor(),
                          transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))                          
                      ])),
            batch_size=args.batch_size, shuffle=True, **kwargs)
        test_loader = torch.utils.data.DataLoader(
            datasets.SVHN(path, train=False,
                      transform=transforms.Compose([
                          #transforms.Resize(48),
                          transforms.CenterCrop(32),
                          transforms.ToTensor(),
                          transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  

                          ])),
            batch_size=args.test_batch_size, shuffle=True, **kwargs)         
    #elif args.dataset == 'CIFAR10':
        #num_class = 10
        #train_loader = torch.utils.data.DataLoader(
            #datasets.CIFAR10(path, train=True, download=True,
                      #transform=transforms.Compose([
                          #transforms.RandomCrop(32, padding=4),
                          #transforms.RandomHorizontalFlip(0.5),                          
                          #transforms.ToTensor(),
                          #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))                          
                      #])),
            #batch_size=args.batch_size, shuffle=True, **kwargs)
        #test_loader = torch.utils.data.DataLoader(
            #datasets.CIFAR10(path, train=False,
                      #transform=transforms.Compose([
                          ##transforms.Resize(48),
                          #transforms.CenterCrop(32),
                          #transforms.ToTensor(),
                          #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))  

                          #])),
            #batch_size=args.test_batch_size, shuffle=True, **kwargs)  
    elif args.dataset == 'CIFAR10':
        num_class = 10
        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(path, train=True, download=True,
                      transform=transforms.Compose([
                          transforms.RandomCrop(32, padding=4),
                          transforms.RandomHorizontalFlip(),                          
                          transforms.ToTensor(),
                          transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2430, 0.2620))                          
                      ])),
            batch_size=args.batch_size, shuffle=True, **kwargs)
        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(path, train=False,
                      transform=transforms.Compose([
                          #transforms.Resize(48),
                          #transforms.CenterCrop(32),
                          transforms.ToTensor(),
                          transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2430, 0.2620))  

                          ])),
            batch_size=args.test_batch_size, shuffle=True, **kwargs)      
    #elif args.dataset == 'CIFAR100':
        #num_class = 100
        #train_loader = torch.utils.data.DataLoader(
            #datasets.CIFAR100(path, train=True, download=True,
                      #transform=transforms.Compose([
                          #transforms.Resize(48),
                          #transforms.RandomCrop(32),
                          #transforms.ColorJitter(brightness=32./255, contrast=0.5),
                          #transforms.ToTensor()
                      #])),
            #batch_size=args.batch_size, shuffle=True, **kwargs)
        #test_loader = torch.utils.data.DataLoader(
            #datasets.CIFAR100(path, train=False,
                      #transform=transforms.Compose([
                          #transforms.Resize(48),
                          #transforms.CenterCrop(32),
                          #transforms.ToTensor()
                      #])),
            #batch_size=args.test_batch_size, shuffle=True, **kwargs)    
    elif args.dataset == 'CIFAR100':
        num_class = 100
        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100(path, train=True, download=True,
                      transform=transforms.Compose([
                          transforms.RandomCrop(32, padding=4),
                          transforms.RandomHorizontalFlip(),
                          transforms.ToTensor(),
                          transforms.Normalize((0.5070, 0.4870, 0.4410), (0.2670, 0.2560, 0.2760)),
                          
                      ])),
            batch_size=args.batch_size, shuffle=True, **kwargs)
        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100(path, train=False,
                      transform=transforms.Compose([
                          transforms.ToTensor(),
                          transforms.Normalize((0.5070, 0.4870, 0.4410), (0.2670, 0.2560, 0.2760)),
                          
                      ])),
            batch_size=args.test_batch_size, shuffle=True, **kwargs)   
    elif args.dataset == 'TinyImageNet':
        num_class = 200
        ########################################################################################
        # train_loader = torch.utils.data.DataLoader(
        #     datasets.ImageNet(path, split='train', download=False,
        #               transform=transforms.Compose([
        #                   transforms.RandomCrop(224),
        #                   transforms.RandomHorizontalFlip(),
        #                   transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        #                   transforms.ToTensor()
        #               ])),
        #     batch_size=args.batch_size, shuffle=True, **kwargs)
        # test_loader = torch.utils.data.DataLoader(
        #     datasets.ImageNet(path, split='val',
        #               transform=transforms.Compose([
        #                   transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        #                   transforms.ToTensor()
        #               ])),
        #     batch_size=args.test_batch_size, shuffle=True, **kwargs) 
        ########################################################################################
        train_data_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])
        ])
        val_data_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])
        ])
        train_dataset = datasets.ImageFolder(
        root='./data/TinyImageNet/train',
        transform=train_data_transform)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4,drop_last=True)

        val_dataset = datasets.ImageFolder(
        root='./data/TinyImageNet/val',
        transform=val_data_transform)
        test_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.test_batch_size, shuffle=True, num_workers=4,drop_last=True)
    elif args.dataset == 'ImageNet':
        num_class = 1000
        ########################################################################################
        # train_loader = torch.utils.data.DataLoader(
        #     datasets.ImageNet(path, split='train', download=False,
        #               transform=transforms.Compose([
        #                   transforms.RandomCrop(224),
        #                   transforms.RandomHorizontalFlip(),
        #                   transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        #                   transforms.ToTensor()
        #               ])),
        #     batch_size=args.batch_size, shuffle=True, **kwargs)
        # test_loader = torch.utils.data.DataLoader(
        #     datasets.ImageNet(path, split='val',
        #               transform=transforms.Compose([
        #                   transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        #                   transforms.ToTensor()
        #               ])),
        #     batch_size=args.test_batch_size, shuffle=True, **kwargs) 
        ########################################################################################
        train_data_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])
        ])
        val_data_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])
        ])
        train_dataset = datasets.ImageFolder(
        root='./imageNet/train',
        transform=train_data_transform)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4,drop_last=True)

        val_dataset = datasets.ImageFolder(
        root='./imageNet/val',
        transform=val_data_transform)
        test_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.test_batch_size, shuffle=True, num_workers=4,drop_last=True)         
    else:
        raise NameError('Undefined dataset {}'.format(args.dataset))
    return num_class, train_loader, test_loader


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


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


def train(train_loader, model, criterion, optimizer, epoch, device):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    
    model.train()
    train_len = len(train_loader)
    epoch_acc = 0
    end = time.time()

    for batch_idx, (data, target) in enumerate(train_loader):
        data_time.update(time.time() - end)

        data, target = data.to(device), target.to(device)
        model = model.to(device)
        optimizer.zero_grad()
        output = model(data)
        r = (1.*batch_idx + (epoch-1)*train_len) / (args.epochs*train_len)
        loss = criterion(output, target, r)
        acc = accuracy(output, target)
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        epoch_acc += acc[0].item()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {}\t[{}/{} ({:.0f}%)]\t'
                  'Loss: {:.6f}\tAccuracy: {:.6f}\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})'.format(
                  epoch, batch_idx * len(data), len(train_loader.dataset),
                  100. * batch_idx / len(train_loader),
                  loss.item(), acc[0].item(),
                  batch_time=batch_time, data_time=data_time))
    return loss, epoch_acc


def snapshot(model, folder, epoch):
    path = os.path.join(folder, 'model_{}.pth'.format(epoch))
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    print('saving model to {}'.format(path))
    torch.save(model.state_dict(), path)


def test(test_loader, model, criterion, device):
    model.eval()
    test_loss = 0
    acc = 0
    test_len = len(test_loader)
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target, r=1).item()
            acc += accuracy(output, target)[0].item()
            # print(output.shape)

    test_loss /= test_len
    acc /= test_len
    print('\nTest set: Average loss: {:.6f}, Accuracy: {:.6f} \n'.format(
        test_loss, acc))
    return test_loss, acc


def main():
    global args, best_prec1
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    device = torch.device(0)
    
    # datasets
    num_class, train_loader, test_loader = get_setting(args)

    # model
    A = 64
     
    A1,A2, B1,B2, = 64, 64, 48, 48 
    C1,C2, D1,D2  = 48,48,48,48
    # A, B, C, D = 32, 32, 32, 32
    model = capsules(A=A,A1= A1,A2= A2, B1=B1,B2=B2, C1=C1, C2=C2, D1=D1, D2=D2, E=num_class,
                     iters=args.em_iters).to(device)
#######################载入权重#########################
    # pretrained_weights = torch.load('./snapshots/model_20.pth',map_location = device)

    # # 检查模型结构是否匹配，如果不匹配，适当调整模型结构
    # model.load_state_dict(pretrained_weights)

    
#######################载入权重#########################
    criterion = SpreadLoss(num_class=num_class, m_min=0.2, m_max=0.9)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=1)

    test_loss, best_acc = test(test_loader, model, criterion, device)
    for epoch in range(1, args.epochs + 1):
        loss, acc = train(train_loader, model, criterion, optimizer, epoch, device)
        loss /= len(train_loader)
        acc /= len(train_loader)
        scheduler.step(acc)
        test_loss1, best_acc1 = test(test_loader, model, criterion, device)
        if epoch % args.test_intvl == 0:
            best_acc = max(best_acc, best_acc1)
            snapshot(model, args.snapshot_folder, epoch)
        text_create('loss-train', 'Epoch: {:.6f}, Average loss: {:.6f}, Accuracy: {:.6f}\n'.format(epoch, loss, acc))
        text_create('loss-acc', 'Epoch: {:.6f}, Average loss: {:.6f}, Accuracy: {:.6f}, Best Accu: {:.6f}\n'.format(epoch, test_loss1, best_acc1, best_acc))
    test_loss1, best_acc1 = test(test_loader, model, criterion, device)
    best_acc = max(best_acc, best_acc1)
    print('best test accuracy: {:.6f}'.format(best_acc))    

if __name__ == '__main__':
    main()
