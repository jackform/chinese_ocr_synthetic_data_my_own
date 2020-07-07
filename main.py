import sys
import torch
import argparse
import numpy as np

from torch.autograd import Variable
from torch.utils import data
import os

from metrics import runningScore
import models
from util import Logger, AverageMeter
import time

from dataloader.rects_data_loader import ReCTSDataLoader


def ohem_single(score, gt_text, training_mask):
    pos_num = (int)(np.sum(gt_text > 0.5)) - (int)(np.sum((gt_text > 0.5) & (training_mask <= 0.5)))

    if pos_num == 0:
        # selected_mask = gt_text.copy() * 0 # may be not good
        selected_mask = training_mask
        selected_mask = selected_mask.reshape(1, selected_mask.shape[0], selected_mask.shape[1]).astype('float32')
        return selected_mask

    neg_num = (int)(np.sum(gt_text <= 0.5))
    neg_num = (int)(min(pos_num * 3, neg_num))

    if neg_num == 0:
        selected_mask = training_mask
        selected_mask = selected_mask.reshape(1, selected_mask.shape[0], selected_mask.shape[1]).astype('float32')
        return selected_mask

    neg_score = score[gt_text <= 0.5]
    neg_score_sorted = np.sort(-neg_score)
    threshold = -neg_score_sorted[neg_num - 1]

    selected_mask = ((score >= threshold) | (gt_text > 0.5)) & (training_mask > 0.5)
    selected_mask = selected_mask.reshape(1, selected_mask.shape[0], selected_mask.shape[1]).astype('float32')
    return selected_mask


def ohem_batch(scores, gt_texts, training_masks):
    scores = scores.data.cpu().numpy()
    gt_texts = gt_texts.data.cpu().numpy()
    training_masks = training_masks.data.cpu().numpy()

    selected_masks = []
    for i in range(scores.shape[0]):
        selected_masks.append(ohem_single(scores[i, :, :], gt_texts[i, :, :], training_masks[i, :, :]))

    selected_masks = np.concatenate(selected_masks, 0)
    selected_masks = torch.from_numpy(selected_masks).float()

    return selected_masks


def dice_loss(input, target, mask):
    input = torch.sigmoid(input)

    input = input.contiguous().view(input.size()[0], -1)
    target = target.contiguous().view(target.size()[0], -1)
    mask = mask.contiguous().view(mask.size()[0], -1)

    input = input * mask
    target = target * mask

    a = torch.sum(input * target, 1)
    b = torch.sum(input * input, 1) + 0.001
    c = torch.sum(target * target, 1) + 0.001
    d = (2 * a) / (b + c)
    dice_loss = torch.mean(d)
    return 1 - dice_loss


def cal_text_score(texts, gt_texts, training_masks, running_metric_text):
    training_masks = training_masks.data.cpu().numpy()
    pred_text = torch.sigmoid(texts).data.cpu().numpy() * training_masks
    pred_text[pred_text <= 0.5] = 0
    pred_text[pred_text > 0.5] = 1
    pred_text = pred_text.astype(np.int32)
    gt_text = gt_texts.data.cpu().numpy() * training_masks
    gt_text = gt_text.astype(np.int32)
    running_metric_text.update(gt_text, pred_text)
    score_text, _ = running_metric_text.get_scores()
    return score_text


def cal_kernel_score(kernels, gt_kernels, gt_texts, training_masks, running_metric_kernel):
    mask = (gt_texts * training_masks).data.cpu().numpy()
    kernel = kernels[:, -1, :, :]
    gt_kernel = gt_kernels[:, -1, :, :]
    pred_kernel = torch.sigmoid(kernel).data.cpu().numpy()
    pred_kernel[pred_kernel <= 0.5] = 0
    pred_kernel[pred_kernel > 0.5] = 1
    pred_kernel = (pred_kernel * mask).astype(np.int32)
    gt_kernel = gt_kernel.data.cpu().numpy()
    gt_kernel = (gt_kernel * mask).astype(np.int32)
    running_metric_kernel.update(gt_kernel, pred_kernel)
    score_kernel, _ = running_metric_kernel.get_scores()
    return score_kernel


def train(train_loader, model, criterion, optimizer, epoch):
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    running_metric_text = runningScore(2)
    running_metric_kernel = runningScore(2)

    end = time.time()
    for batch_idx, (imgs, gt_texts, gt_kernels, training_masks) in enumerate(train_loader):
        data_time.update(time.time() - end)

        # imgs = Variable(imgs.cuda())
        # gt_texts = Variable(gt_texts.cuda())
        # gt_kernels = Variable(gt_kernels.cuda())
        # training_masks = Variable(training_masks.cuda())
        imgs = Variable(imgs)
        gt_texts = Variable(gt_texts)
        gt_kernels = Variable(gt_kernels)
        training_masks = Variable(training_masks)

        outputs = model(imgs)
        texts = outputs[:, 0, :, :]
        kernels = outputs[:, 1:, :, :]

        selected_masks = ohem_batch(texts, gt_texts, training_masks)
        # selected_masks = Variable(selected_masks.cuda())
        selected_masks = Variable(selected_masks)

        loss_text = criterion(texts, gt_texts, selected_masks)

        loss_kernels = []
        mask0 = torch.sigmoid(texts).data.cpu().numpy()
        mask1 = training_masks.data.cpu().numpy()
        selected_masks = ((mask0 > 0.5) & (mask1 > 0.5)).astype('float32')
        selected_masks = torch.from_numpy(selected_masks).float()
        # selected_masks = Variable(selected_masks.cuda())
        selected_masks = Variable(selected_masks)
        for i in range(6):
            kernel_i = kernels[:, i, :, :]
            gt_kernel_i = gt_kernels[:, i, :, :]
            loss_kernel_i = criterion(kernel_i, gt_kernel_i, selected_masks)
            loss_kernels.append(loss_kernel_i)
        loss_kernel = sum(loss_kernels) / len(loss_kernels)

        loss = 0.7 * loss_text + 0.3 * loss_kernel
        losses.update(loss.item(), imgs.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        score_text = cal_text_score(texts, gt_texts, training_masks, running_metric_text)
        score_kernel = cal_kernel_score(kernels, gt_kernels, gt_texts, training_masks, running_metric_kernel)

        batch_time.update(time.time() - end)
        end = time.time()

        # if batch_idx % 20 == 0:
        if True:
            output_log = '({batch}/{size}) Batch: {bt:.3f}s | TOTAL: {total:.0f}min | ETA: {eta:.0f}min | Loss: {loss:.4f} | Acc_t: {acc: .4f} | IOU_t: {iou_t: .4f} | IOU_k: {iou_k: .4f}'.format(
                batch=batch_idx + 1,
                size=len(train_loader),
                bt=batch_time.avg,
                total=batch_time.avg * batch_idx / 60.0,
                eta=batch_time.avg * (len(train_loader) - batch_idx) / 60.0,
                loss=losses.avg,
                acc=score_text['Mean Acc'],
                iou_t=score_text['Mean IoU'],
                iou_k=score_kernel['Mean IoU'])
            print(output_log)
            sys.stdout.flush()

    return (
        losses.avg, score_text['Mean Acc'], score_kernel['Mean Acc'], score_text['Mean IoU'], score_kernel['Mean IoU'])


def adjust_learning_rate(schedule, lr, optimizer, epoch):
    global state
    if epoch in schedule:
        lr = lr * 0.1
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    return lr


def save_checkpoint(state, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)


def main():
    # parser = argparse.ArgumentParser(description='Hyperparams')
    # parser.add_argument('--arch', nargs='?', type=str, default='resnet50')
    # parser.add_argument('--img_size', nargs='?', type=int, default=640,
    #                     help='Height of the input image')
    # parser.add_argument('--n_epoch', nargs='?', type=int, default=600,
    #                     help='# of the epochs')
    # parser.add_argument('--schedule', type=int, nargs='+', default=[200, 400],
    #                     help='Decrease learning rate at these epochs.')
    # parser.add_argument('--batch_size', nargs='?', type=int, default=1,
    #                     help='Batch Size')
    # parser.add_argument('--lr', nargs='?', type=float, default=1e-3,
    #                     help='Learning Rate')
    # parser.add_argument('--resume', nargs='?', type=str, default=None,
    #                     help='Path to previous saved model to restart from')
    # parser.add_argument('--checkpoint', default='', type=str, metavar='PATH',
    #                     help='path to save checkpoint (default: checkpoint)')
    # args = parser.parse_args()

    # lr = args.lr
    # schedule = args.schedule
    # batch_size = args.batch_size
    # n_epoch = args.n_epoch
    # image_size = args.img_size
    # resume = args.resume
    # checkpoint_path = args.checkpoint
    # arch = args.arch

    lr = 1e-3
    schedule = [200, 400]
    batch_size = 1
    n_epoch = 600
    image_size = 640
    resume = None
    checkpoint_path = ''
    arch = 'resnet50'

    if checkpoint_path == '':
        checkpoint_path = "checkpoints/ctw1500_%s_bs_%d_ep_%d" % (arch, batch_size, n_epoch)

    print('checkpoint path: %s' % checkpoint_path)
    print('init lr: %.8f' % lr)
    print('schedule: ', schedule)
    sys.stdout.flush()

    if not os.path.isdir(checkpoint_path):
        os.makedirs(checkpoint_path)

    kernel_num = 7
    min_scale = 0.4
    start_epoch = 0

    data_loader = ReCTSDataLoader(need_transform=True,
                                  img_size=image_size,
                                  kernel_num=kernel_num,
                                  min_scale=min_scale,
                                  train_data_dir='data/ReCTS/img/',
                                  train_gt_dir='data/ReCTS/gt/'
                                  )

    ctw_root_dir = 'data/'

    train_loader = torch.utils.data.DataLoader(
        data_loader,
        batch_size=batch_size,
        shuffle=True,
        num_workers=3,
        drop_last=True,
        pin_memory=True)

    if arch == "resnet50":
        model = models.resnet50(pretrained=False, num_classes=kernel_num)
    elif arch == "resnet101":
        model = models.resnet101(pretrained=False, num_classes=kernel_num)
    elif arch == "resnet152":
        model = models.resnet152(pretrained=False, num_classes=kernel_num)

    # model = torch.nn.DataParallel(model).cuda()
    model = torch.nn.DataParallel(model)

    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.99, weight_decay=5e-4)

    title = 'ReCTS'
    if resume:
        print('Resuming from checkpoint.')
        assert os.path.isfile(resume), 'Error: no checkpoint directory found!'
        checkpoint = torch.load(resume)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        logger = Logger(os.path.join(checkpoint_path, 'log.txt'), title=title, resume=True)
    else:
        print('Training from scratch.')
        logger = Logger(os.path.join(checkpoint_path, 'log.txt'), title=title)
        logger.set_names(['Learning Rate', 'Train Loss', 'Train Acc.', 'Train IOU.'])

    for epoch in range(start_epoch, n_epoch):
        lr = adjust_learning_rate(schedule, lr, optimizer, epoch)
        print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, n_epoch, optimizer.param_groups[0]['lr']))

        train_loss, train_te_acc, train_ke_acc, train_te_iou, train_ke_iou = train(train_loader, model, dice_loss,
                                                                                   optimizer, epoch)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'lr': lr,
            'optimizer': optimizer.state_dict(),
        }, checkpoint=checkpoint_path)

        logger.append([optimizer.param_groups[0]['lr'], train_loss, train_te_acc, train_te_iou])
    logger.close()


if __name__ == '__main__':
    main()

# parser = argparse.ArgumentParser(description='Hyperparams')
