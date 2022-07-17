import random
import time
import warnings
import argparse
import shutil
import os.path as osp
from pprint import pprint

##
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
import torchvision.transforms as T
import torch.nn.functional as F

####
import datasets
import models
from models.domain_discriminator import DomainDiscriminator
from utils.dann import DomainAdversarialLoss, ImageClassifier
from utils.data import ForeverDataIterator, ResizeImage
from utils.metric import accuracy, ConfusionMatrix
from utils.meter import AverageMeter, ProgressMeter
from utils.logger import CompleteLogger
from utils.analysis import collect_feature, tsne, a_distance
from utils.mbank import memoryBank
from  utils.loss import DynamicSmooth, CrossEntropyLabelSmooth, SemanticLoss, MI
from utils.cdan import ConditionalDomainAdversarialLoss


log_info = dict()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def opts():

    architecture_names = sorted(name for name in models.__dict__
                                if name.islower() and not name.startswith("__") and callable(models.__dict__[name]))
    dataset_names = sorted(name for name in datasets.__dict__
                                if not name.startswith("__") and callable(datasets.__dict__[name]))
    parser = argparse.ArgumentParser(description='DANN or CDAN for Unsupervised Domain Adaptation')

    # dataset parameters
    parser.add_argument('root', metavar='DIR', default='office31',
                        help='root path of dataset')
    parser.add_argument('-d', '--data', metavar='DATA', default='Office31',
                        help='dataset: ' + ' | '.join(dataset_names) +
                             ' (default: Office31)')
    parser.add_argument('-s', '--source', default='A', help='source domain(s)')
    parser.add_argument('-t', '--target', default='W', help='target domain(s)')
    parser.add_argument('--center-crop', default=False, action='store_true', ## VISDA-2017: True
                        help='whether use center crop during training')

    # model parameters
    parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                        choices=architecture_names,
                        help='backbone architecture: ' +
                             ' | '.join(architecture_names) +
                             ' (default: resnet50)')
    parser.add_argument('--bottleneck-dim', default=256, type=int,
                        help='Dimension of bottleneck')
    parser.add_argument('--base', default='cdan', type=str, help='baseline') ## dann or cdan
    ### cdan parameters
    parser.add_argument('-r', '--randomized', default=False, type=bool,
                        help='using randomized multi-linear-map (default: False)')
    parser.add_argument('-rd', '--randomized-dim', default=1024, type=int,
                        help='randomized dimension when using randomized multi-linear-map (default: 1024)')
    parser.add_argument('--entropy', default=True, action='store_true', help='use entropy conditioning')

    ### training parameters
    parser.add_argument('--trade-off', default=1., type=float,
                        help='the trade-off hyper-parameter for transfer loss')
    parser.add_argument('--trade-off1', default=0.2, type=float,
                        help='the trade-off hyper-parameter for transfer loss')
    parser.add_argument('--trade-off2', default=0.08, type=float,
                        help='the trade-off hyper-parameter for transfer loss')
    parser.add_argument('-b', '--batch-size', default=32, type=int,
                        metavar='N',
                        help='mini-batch size (default: 32)')
    parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--lr-gamma', default=0.001, type=float, help='parameter for lr scheduler')
    parser.add_argument('--lr-decay', default=0.75, type=float, help='parameter for lr scheduler')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay',default=1e-3, type=float,
                        metavar='W', help='weight decay (default: 1e-3)',
                        dest='weight_decay')

    #### log parameters
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=20, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-i', '--iters-per-epoch', default=1000, type=int,
                        help='Number of iterations per epoch')
    parser.add_argument('-p', '--print-freq', default=100, type=int,
                        metavar='N', help='print frequency (default: 100)')
    parser.add_argument('--seed', default=2, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--per-class-eval', action='store_true',
                        help='whether output per-class accuracy during evaluation')
    parser.add_argument("--log", type=str, default='logs1/lst_dann1/office31_A2W',
                        help="Where to save logs1, checkpoints and debugging images.")
    parser.add_argument("--phase", type=str, default='train', choices=['train', 'test', 'analysis'],
                        help="When phase is 'test', only test the model."
                             "When phase is 'analysis', only analysis the model.")
    args = parser.parse_args()

    args.root =  '/disks/disk0/feifei/paper/paper3-3090/' + args.root ###

    return args


def main():

    args = opts()
    logger = CompleteLogger(args.log, args.phase)
    pprint(args)
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')
    cudnn.benchmark = True
    # Data loading code
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    if args.center_crop:
        train_transform = T.Compose([
            ResizeImage(256),
            T.CenterCrop(224),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            normalize
        ])
    else:
        train_transform = T.Compose([
            ResizeImage(256),
            T.RandomResizedCrop(224),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            normalize
        ])

    val_transform = T.Compose([
        ResizeImage(256),
        T.CenterCrop(224),
        T.ToTensor(),
        normalize
    ])

    dataset = datasets.__dict__[args.data]
    train_source_dataset = dataset(root=args.root, task=args.source, download=True, transform=train_transform)
    train_source_loader = DataLoader(train_source_dataset, batch_size=args.batch_size,
                                     shuffle=True, num_workers=args.workers, drop_last=True)
    train_target_dataset = dataset(root=args.root, task=args.target, download=True, transform=train_transform)
    train_target_loader = DataLoader(train_target_dataset, batch_size=args.batch_size,
                                     shuffle=True, num_workers=args.workers, drop_last=True)
    val_dataset = dataset(root=args.root, task=args.target, download=True, transform=val_transform)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    if args.data == 'DomainNet':
        test_dataset = dataset(root=args.root, task=args.target, split='test', download=True, transform=val_transform)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    else:
        test_loader = val_loader
    train_source_iter = ForeverDataIterator(train_source_loader)
    train_target_iter = ForeverDataIterator(train_target_loader)
    # create model
    print("=> using pre-trained model '{}'".format(args.arch))
    backbone = models.__dict__[args.arch](pretrained=True)
    classifier = ImageClassifier(backbone, train_source_dataset.num_classes, bottleneck_dim=args.bottleneck_dim).to(device)

    classifier_feature_dim = classifier.features_dim
    num_classes = train_source_dataset.num_classes
    ###### chose baseline: dann or cdan
    if args.base == 'dann':
        domain_discri = DomainDiscriminator(in_feature=classifier_feature_dim, hidden_size=1024).to(device) # 定义域判别器
        domain_adv = DomainAdversarialLoss(domain_discri).to(device)
    else: #CDAN
        if args.randomized:
            domain_discri = DomainDiscriminator(args.randomized_dim, hidden_size=1024).to(device)
            domain_adv = ConditionalDomainAdversarialLoss(domain_discri, entropy_conditioning=args.entropy,
                                                          num_classes=num_classes, features_dim=classifier_feature_dim,
                                                          randomized=args.randomized,
                                                          randomized_dim=args.randomized_dim).to(device)
        else:
            domain_discri = DomainDiscriminator(classifier_feature_dim * num_classes, hidden_size=1024).to(device)
            domain_adv = ConditionalDomainAdversarialLoss(domain_discri, entropy_conditioning=args.entropy,
                                                          num_classes=num_classes, features_dim=classifier_feature_dim,
                                                          randomized=args.randomized,
                                                          randomized_dim=args.randomized_dim).to(device)
    ##### init memory bank
    bank = memoryBank(num_classes, classifier_feature_dim)
    criterion = DynamicSmooth(num_classes)
    ### semantic transport loss
    semanticLoss = SemanticLoss(num_classes).to(device)
    # define optimizer and lr scheduler
    optimizer = SGD(classifier.get_parameters() + domain_discri.get_parameters(),
                    args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)
    lr_scheduler = LambdaLR(optimizer, lambda x:  args.lr * (1. + args.lr_gamma * float(x)) ** (-args.lr_decay))

    # # resume from the best checkpoint
    if args.phase != 'train':
        checkpoint = torch.load(logger.get_checkpoint_path('best'), map_location='cpu')
        classifier.load_state_dict(checkpoint)
    # analysis the model
    if args.phase == 'analysis':
        # extract features from both domains
        feature_extractor = nn.Sequential(classifier.backbone, classifier.bottleneck).to(device)
        source_feature = collect_feature(train_source_loader, feature_extractor, device)
        target_feature = collect_feature(train_target_loader, feature_extractor, device)
        # plot t-SNE
        tSNE_filename = osp.join(logger.visualize_directory, 'TSNE.png')
        # tSNE_filenam = osp.join(logger.visualize_directory, 'TSNE.pdf')
        tsne.visualize(source_feature, target_feature, tSNE_filename)
        print("Saving t-SNE to", tSNE_filename)
        # calculate A-distance, which is a measure for distribution discrepancy
        # A_distance = a_distance.calculate(source_feature, target_feature, device)
        # print("A-distance =", A_distance)
        return

    if args.phase == 'test':
        acc1 = validate(test_loader, classifier, args)
        print(acc1)
        return

    # start training

    best_acc1 = 0.
    for epoch in range(args.epochs):
        # train for one epoch
        train(train_source_iter, train_target_iter, classifier, domain_adv, semanticLoss, bank, criterion,
              optimizer, lr_scheduler, epoch, args)
        # evaluate on validation set
        acc1 = validate(val_loader, classifier, args)
        # remember best acc@1 and save checkpoint
        torch.save(classifier.state_dict(), logger.get_checkpoint_path('latest'))
        if acc1 > best_acc1:
            shutil.copy(logger.get_checkpoint_path('latest'), logger.get_checkpoint_path('best'))
        best_acc1 = max(acc1, best_acc1)
    print("best_acc1 = {:3.1f}".format(best_acc1))
    # evaluate on test set
    classifier.load_state_dict(torch.load(logger.get_checkpoint_path('best')))
    acc1 = validate(test_loader, classifier, args)
    print("test_acc1 = {:3.1f}".format(acc1))
    logger.close()


def train(train_source_iter, train_target_iter, model, domain_adv, semanticLoss, bank, criterion,
          optimizer, lr_scheduler, epoch, args):

    batch_time = AverageMeter('Time', ':5.2f')
    data_time = AverageMeter('Data', ':5.2f')
    losses = AverageMeter('Loss', ':6.2f')
    cls_losses = AverageMeter('cls_losses', ':3.2f')
    transfer_losses = AverageMeter('transfer_loss', ':3.2f')
    semlosses = AverageMeter('semantic loss', ':3.2f')
    cls_accs = AverageMeter('Cls Acc', ':3.1f')
    MI_losses = AverageMeter('MI_loss', ':3.1f')
    domain_accs = AverageMeter('Domain Acc', ':3.1f')
    progress = ProgressMeter(args.iters_per_epoch,
                    [losses, cls_losses, transfer_losses, semlosses, MI_losses, cls_accs, domain_accs],
                    prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()
    domain_adv.train()
    semanticLoss.train()
    criterion.train()
    end = time.time()
    for i in range(args.iters_per_epoch):
        x_s, labels_s = next(train_source_iter)
        x_t, _ = next(train_target_iter)
        x_s = x_s.to(device)
        x_t = x_t.to(device)
        labels_s = labels_s.to(device)
        # measure data loading time
        data_time.update(time.time() - end)
        # compute output
        x = torch.cat((x_s, x_t), dim=0)
        y, f = model(x)
        y_s, y_t = y.chunk(2, dim=0)
        f_s, f_t = f.chunk(2, dim=0)
        ## predict
        y_s_softmax = F.softmax(y_s, dim=1)  #
        _, predict_s = torch.max(y_s_softmax, dim=1)
        #####
        de_feats, de_ys = bank.get()  ##dequeue
        center_s, smooth_rate = criterion(f_s, labels_s, de_feats, de_ys, epoch)
        his_y_s = bank.cum(labels_s, y_s_softmax)  # calculate soft predition
        smooth_rate = smooth_rate.to(device)

        ######
        # cls_loss = F.cross_entropy(y_s, labels_s) # hard label
        ## label smooth: type='dynamic' or 'vanilla'
        cls_loss = CrossEntropyLabelSmooth(num_classes=model.num_classes, epsilon=smooth_rate, eps_vanilla = 0.1,
                                           reduction='mean', type='dynamic')(y_s, labels_s, his_y_s=his_y_s)
        semloss = semanticLoss(center_s.detach(), f_t) ##semantic transport
        if args.base == 'dann':
            transfer_loss = domain_adv(f_s, f_t)
        else:
            transfer_loss = domain_adv(y_s, f_s, y_t, f_t)
        MI_loss = MI(y_t)
        loss = cls_loss + transfer_loss * args.trade_off  + semloss*args.trade_off1 - MI_loss*args.trade_off2
        ##### enqueue
        correct_mask = torch.where(predict_s == labels_s)
        if len(correct_mask[0]) !=0:
            enqueue_feats = f_s[correct_mask]
            enqueue_targets = labels_s[correct_mask]
            enqueue_y_s = y_s_softmax[correct_mask]
            bank.enqueue_dequeue(enqueue_feats.detach(), enqueue_targets.detach(), enqueue_y_s.detach())

        #############################################################
        domain_acc = domain_adv.domain_discriminator_accuracy
        cls_acc = accuracy(y_s, labels_s)[0]
        losses.update(loss.item(), x_s.size(0))
        cls_losses.update(cls_loss.item(), x_s.size(0))
        semlosses.update(semloss.item(), x_s.size(0))
        transfer_losses.update(transfer_loss.item(), x_s.size(0))
        cls_accs.update(cls_acc.item(), x_s.size(0))
        MI_losses.update(MI_loss.item(), x_s.size(0))
        domain_accs.update(domain_acc.item(), x_s.size(0))
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if i % args.print_freq == 0:
            progress.display(i)


def validate(val_loader: DataLoader, model: ImageClassifier, args: argparse.Namespace) -> float:
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')
    # switch to evaluate mode
    model.eval()
    if args.per_class_eval:
        classes = val_loader.dataset.classes
        confmat = ConfusionMatrix(len(classes))
    else:
        confmat = None
    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            images = images.to(device)
            target = target.to(device)
            # compute output
            output, _ = model(images)
            loss = F.cross_entropy(output, target)
            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            if confmat:
                confmat.update(target, output.argmax(1))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1.item(), images.size(0))
            top5.update(acc5.item(), images.size(0))
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            if i % args.print_freq == 0:
                progress.display(i)
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))
        if confmat:
            print(confmat.format(classes))
    return top1.avg


if __name__ == '__main__':

    main()

