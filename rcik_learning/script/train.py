import os
import torch
import argparse
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import data_loader as dl
import model as ml
from utils import GetRankingAccuracy, GetRankingAccuracyEqualCase
from loss import RankingLoss, DenseRankingLoss
from tqdm import tqdm

parser = argparse.ArgumentParser(description='RCIK Learning')
parser.add_argument('--epochs', type=int, default=100, metavar='N', help='number of epochs to train (default: 100)')

# basic settings
parser.add_argument('--name', default='fetch_new', type=str, help='output model name')
parser.add_argument('--gpu_ids', default='0', type=str, help='gpu_ids: e.g. 0  0,1,2  0,2')

# robot -> 0: iiwa, 1: fetch, 2: fetch_8dof
parser.add_argument('--mode', default=1, type=int, help='model selection')

# hyper-param
parser.add_argument('--batchsize', default=128, type=int, help='batchsize')
parser.add_argument('--lr', type=float, default=5e-5, metavar='LR', help='learning rate (default: 5e-5)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='LR', help=' ')
parser.add_argument('--mu', type=float, default=10.0, metavar='weight', help=' ')

# log
parser.add_argument('--save_dir', type=str, default='./models', help='directory for model saving')
parser.add_argument('--dataset_dir', type=str, default='/media/mincheul/db/rrr/rcik_dataset_2022/', help='dataset directory')

step = 0
opts = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_ids

def main():
    np.random.seed(123)
    torch.manual_seed(123)

    if (opts.mode == 0):
        dirpath = opts.dataset_dir + 'iiwa/'
        DOF = 7
        isRobotBody = False
    elif (opts.mode == 1):
        dirpath = opts.dataset_dir + 'fetch_arm/'
        DOF = 7
        isRobotBody = True
    else:
        dirpath = opts.dataset_dir + 'fetch_8dof/'
        DOF = 8
        isRobotBody = True

    train_dataset = dl.TrainDataset(dirpath, DOF, isRobotBody=isRobotBody)
    val_dataset = dl.ValidationDataset(dirpath, DOF, 200, 5000, isRobotBody=isRobotBody)

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=128, shuffle=True, num_workers=0, pin_memory=True)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=32, shuffle=False, num_workers=0)

    model = ml.RCIK(DOF)
    # checkpoint = torch.load("./models/fetch_7dof.pt")
    model = model.cuda()
    model.train()
    # model.load_state_dict(checkpoint)

    # state = model.state_dict()
    # torch.save(state, os.path.join(opts.save_dir, 'fetch_7dof_best.pt'))

    ''' hyper-param for traning'''
    # criterion = torch.nn.L1Loss()
    criterion = torch.nn.MSELoss()
    ranking_criterion = RankingLoss(margin=0.0)

    optimizer = torch.optim.Adam(model.parameters(), lr=opts.lr)
    # optimizer = torch.optim.SGD(model.parameters(), lr=opts.lr, momentum=opts.momentum, nesterov=True)

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 100], gamma=0.1)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, opts.epochs - opts.warm)

    ''' for recording'''
    writer = SummaryWriter("./logs/" + opts.name)

    with open(os.path.join(opts.save_dir, opts.name + '.txt'), "w") as f:
        for idx, m in enumerate(model.named_modules()):
            for i in m:
                f.write(str(i))

    best = 0
    print('start training.')
    for epoch in range(1, opts.epochs + 1):
        mse_loss, ranking_loss = train_epoch(opts, model, train_dataloader, criterion,
                                             ranking_criterion, optimizer, epoch)
        scheduler.step(epoch)

        '''Train Step'''
        total_loss = mse_loss + ranking_loss * opts.mu
        print('Epoch:[{}/{}] total loss:{:.4f} mse loss:{:.4f} ranking loss:{:.4f}'.format( \
            epoch, opts.epochs, total_loss, mse_loss, ranking_loss))

        '''Valid Step'''
        val_acc = validation_accuracy(model, val_dataloader)
        print('Epoch:[{}/{}] validation accuracy:{:.4f}'.format(epoch, opts.epochs, val_acc))

        '''Log Step'''
        writer.add_scalar('Total Loss', mse_loss + ranking_loss, epoch)
        writer.add_scalar('MSE Loss', mse_loss, epoch)
        writer.add_scalar('Ranking Loss', ranking_loss, epoch)
        writer.add_scalar('Ranking Accuracy', val_acc, epoch)

        '''Save Step'''
        if val_acc > best:
            best = val_acc
            state = model.state_dict()
            torch.save(state, os.path.join(opts.save_dir, opts.name + '_best.pt'))

        if epoch % 1 == 0:
            state = model.state_dict()
            torch.save(state, os.path.join(opts.save_dir, opts.name + '_e{}.pt'.format(epoch)))

    writer.close()


def train_epoch(opts, model, loader, criterion, ranking_criterion, optimizer, epoch):
    mse_losses = 0.
    ranking_losses = 0.
    nCut = 0.
    model.train()

    for i, data in enumerate(tqdm(loader)):
        x, x_wsi, y = data
        x, x_wsi, y = x.cuda(), x_wsi.cuda(), y.cuda()

        f_wsi = model.get_env_feature(x_wsi)
        # preds = model(x, x_wsi)
        preds = model(x, f_wsi)

        mse_loss = criterion(preds, y)
        ranking_loss = ranking_criterion(preds, y)
        loss = mse_loss + ranking_loss * opts.mu

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        mse_losses += mse_loss.item() * x.size(0)
        ranking_losses += ranking_loss.item() * x.size(0)
        nCut += x.size(0)
    return mse_losses / nCut, ranking_losses / nCut


def validation_accuracy(model, loader):
    model.eval()
    with torch.no_grad():
        acc = 0
        for i, data in enumerate(loader):
            x, x_wsi, y = data
            x, x_wsi, y = x.cuda(), x_wsi.cuda(), y.cuda()

            env = x_wsi[0].repeat(x.size(0), 1, 1, 1, 1)
            f_wsi = model.get_env_feature(env)
            # preds = model(x, x_wsi[0])  # Should pass single env map for validation
            preds = model(x, f_wsi)  # Should pass single env map for validation

            acc += GetRankingAccuracy(preds, y)
            # acc += GetRankingAccuracyEqualCase(preds, rank_y[i], 50)

    return acc / len(loader)

if __name__ == '__main__':
    main()
