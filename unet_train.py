from model.unet.unet_model import UNet
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from torch.utils.data import DataLoader
from util.dataset import DukeLiverDatasetSegmentation
from torch import optim
import torch.nn as nn
import torch
import logging
import torch.nn.functional as F
from util.dice_loss import dice_coeff
from torchvision.transforms import transforms

def eval_net(net, loader, device):
    """Evaluation without the densecrf with the dice coefficient"""
    net.eval()
    mask_type = torch.float32 if net.n_classes == 1 else torch.long
    n_val = len(loader)  # the number of batch
    tot = 0

    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for batch in loader:
            imgs, true_masks = batch['image'], batch['mask']
            imgs = imgs.to(device=device, dtype=torch.float32)
            true_masks = true_masks.to(device=device, dtype=mask_type)

            with torch.no_grad():
                mask_pred = net(imgs)

            if net.n_classes > 1:
                tot += F.cross_entropy(mask_pred, true_masks).item()
            else:
                pred = torch.sigmoid(mask_pred)
                pred = (pred > 0.5).float()
                tot += dice_coeff(pred, true_masks).item()
            pbar.update()

    net.train()
    return tot / n_val


def train_unet(model,img_dir_train, mask_dir_train, img_dir_val, mask_dir_val,device , batchsize,save_cp, saved_model_dir, lr, epochs):
    '''
    Train a unet model
    '''
    T = transforms.Resize((512, 512))
    dataset_train = DukeLiverDatasetSegmentation(img_dir=img_dir_train,mask_dir=mask_dir_train,
                                                 img_transform=T, mask_transform=T)
    dataset_val = DukeLiverDatasetSegmentation(img_dir=img_dir_val,mask_dir=mask_dir_val,
                                               img_transform=T, mask_transform=T)
    train_loader = DataLoader(dataset_train, batch_size=batchsize, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(dataset_val, batch_size=batchsize, shuffle=False, num_workers=0, pin_memory=True,
                            drop_last=True)

    optimizer = optim.RMSprop(model.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min' if model.n_classes > 1 else 'max', patience=2)
    if model.n_classes > 1:
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.BCEWithLogitsLoss()

    writer = SummaryWriter(comment=f'LR_{lr}_BS_{batchsize}')
    n_train = len(dataset_train)
    n_val = len(dataset_val)
    sample_total = n_train+n_val
    global_step=0
    for epoch in range(epochs):
        model.train()

        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for imgs, true_masks in train_loader:
                assert imgs.shape[1] == model.n_channels, \
                    f'Network has been defined with {model.n_channels} input channels, ' \
                    f'but loaded images have {imgs.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                imgs = imgs.to(device=device, dtype=torch.float32)
                mask_type = torch.float32 if model.n_classes == 1 else torch.long
                true_masks = true_masks.to(device=device, dtype=mask_type)

                masks_pred = model(imgs)
                loss = criterion(masks_pred, true_masks)
                epoch_loss += loss.item()
                writer.add_scalar('Loss/train', loss.item(), global_step)

                pbar.set_postfix(**{'loss (batch)': loss.item()})

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_value_(model.parameters(), 0.1)
                optimizer.step()

                pbar.update(imgs.shape[0])
                global_step += 1
                if global_step % (sample_total // (2 * batchsize)) == 0:
                    for tag, value in model.named_parameters():
                        tag = tag.replace('.', '/')
                        writer.add_histogram('weights/' + tag, value.data.cpu().numpy(), global_step)
                        writer.add_histogram('grads/' + tag, value.grad.data.cpu().numpy(), global_step)
                    val_score = eval_net(model, val_loader, device)
                    scheduler.step(val_score)
                    writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], global_step)

                    if model.n_classes > 1:
                        logging.info('Validation cross entropy: {}'.format(val_score))
                        writer.add_scalar('Loss/test', val_score, global_step)
                    else:
                        logging.info('Validation Dice Coeff: {}'.format(val_score))
                        writer.add_scalar('Dice/test', val_score, global_step)

                    writer.add_images('images', imgs, global_step)
                    if model.n_classes == 1:
                        writer.add_images('masks/true', true_masks, global_step)
                        writer.add_images('masks/pred', torch.sigmoid(masks_pred) > 0.5, global_step)

        if save_cp:
            torch.save(model.state_dict(),
                       saved_model_dir + f"_{epoch}.pth")
            logging.info(f'Checkpoint saved !')


def demo():
    img_dir_train = "/home/data/duke_liver/duke_liver_coco/imgs/t1nfs"
    mask_dir_train = "/home/data/duke_liver/duke_liver_coco/masks/t1nfs"
    img_dir_val = "/home/data/duke_liver/duke_liver_coco/imgs/fat"
    mask_dir_val = "/home/data/duke_liver/duke_liver_coco/masks/fat"
    saved_model_dir = "/home/data/duke_liver/results/20210803_unet"
    lr = 0.01
    epochs = 50
    batchsize = 16
    save_cp = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



    model = UNet(n_channels=1, n_classes=1, bilinear=True)
    train_unet(model=model,
               img_dir_train=img_dir_train,
               mask_dir_train=mask_dir_train,
               img_dir_val=img_dir_val,
               mask_dir_val=mask_dir_val,
               device=device,
               save_cp= save_cp,
               batchsize=batchsize,
               saved_model_dir=saved_model_dir,
               lr=lr,
               epochs=epochs)


if __name__ == "__main__":
    demo()
