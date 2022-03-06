import torch
from losses.semantic_loss import CE_Loss, Dice_loss, Focal_Loss
from losses.grad_loss import GRAD_LOSS
from tqdm import tqdm

from utils.utils import get_lr
from utils.utils_metrics import f_score


def fit_one_epoch(model_train, model, loss_history, optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val, Epoch,
                  cuda, dice_loss, focal_loss, cls_weights, num_classes):
    total_loss = 0
    total_f_score = 0

    val_loss = 0
    val_f_score = 0

    model_train.train()
    print('Start Train')
    with tqdm(total=epoch_step, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3) as pbar:
        for iteration, batch in enumerate(gen):
            if iteration >= epoch_step:
                break
            imgs, pngs, labels, depths = batch

            with torch.no_grad():
                imgs = torch.from_numpy(imgs).type(torch.FloatTensor)
                pngs = torch.from_numpy(pngs).long()
                labels = torch.from_numpy(labels).type(torch.FloatTensor)
                depths = torch.from_numpy(depths).type(torch.FloatTensor)
                depths = torch.unsqueeze(depths, 1)
                weights = torch.from_numpy(cls_weights)
                if cuda:
                    imgs = imgs.cuda()
                    pngs = pngs.cuda()
                    labels = labels.cuda()
                    depths = depths.cuda()
                    weights = weights.cuda()

            optimizer.zero_grad()

            sem_outputs, depth_outputs = model_train(imgs)

            # 语义分割部分的损失项计算
            if focal_loss:
                sem_loss = Focal_Loss(sem_outputs, pngs, weights, num_classes=num_classes)
            else:
                sem_loss = CE_Loss(sem_outputs, pngs, weights, num_classes=num_classes)

            if dice_loss:
                main_dice = Dice_loss(sem_outputs, labels)
                sem_loss = sem_loss + main_dice

            with torch.no_grad():
                # -------------------------------#
                #   计算f_score
                # -------------------------------#
                _f_score = f_score(sem_outputs, labels)

            # 深度值损失部分
            depth_loss = GRAD_LOSS(depths, depth_outputs)

            loss = sem_loss + depth_loss

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_f_score += _f_score.item()

            pbar.set_postfix(**{'total_loss': total_loss / (iteration + 1),
                                'depth_loss': depth_loss.item() / (iteration + 1),
                                'f_score': total_f_score / (iteration + 1),
                                'lr': get_lr(optimizer)})
            pbar.update(1)

    print('Finish Train')

    model_train.eval()
    print('Start Validation')
    with tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3) as pbar:
        for iteration, batch in enumerate(gen_val):
            if iteration >= epoch_step_val:
                break
            imgs, pngs, labels, depths = batch
            with torch.no_grad():
                imgs = torch.from_numpy(imgs).type(torch.FloatTensor)
                pngs = torch.from_numpy(pngs).long()
                labels = torch.from_numpy(labels).type(torch.FloatTensor)
                depths = torch.from_numpy(depths).type(torch.FloatTensor)
                depths = torch.unsqueeze(depths, 1)
                weights = torch.from_numpy(cls_weights)
                if cuda:
                    imgs = imgs.cuda()
                    pngs = pngs.cuda()
                    labels = labels.cuda()
                    depths = depths.cuda()
                    weights = weights.cuda()

                sem_outputs, depth_outputs = model_train(imgs)
                if focal_loss:
                    sem_loss = Focal_Loss(sem_outputs, pngs, weights, num_classes=num_classes)
                else:
                    sem_loss = CE_Loss(sem_outputs, pngs, weights, num_classes=num_classes)

                if dice_loss:
                    main_dice = Dice_loss(sem_outputs, labels)
                    sem_loss = sem_loss + main_dice
                # -------------------------------#
                #   计算f_score
                # -------------------------------#
                _f_score = f_score(sem_outputs, labels)

                # 深度值损失部分
                depth_loss = GRAD_LOSS(depths, depth_outputs)

                loss = sem_loss + depth_loss

                val_loss += loss.item()
                val_f_score += _f_score.item()

            pbar.set_postfix(**{'total_loss': val_loss / (iteration + 1),
                                'depth_loss': depth_loss.item() / (iteration + 1),
                                'f_score': val_f_score / (iteration + 1),
                                'lr': get_lr(optimizer)})
            pbar.update(1)

    loss_history.append_loss(total_loss / (epoch_step + 1), val_loss / (epoch_step_val + 1))
    print('Finish Validation')
    print('Epoch:' + str(epoch + 1) + '/' + str(Epoch))
    print('Total Loss: %.3f || Val Loss: %.3f ' % (total_loss / (epoch_step + 1), val_loss / (epoch_step_val + 1)))
    torch.save(model.state_dict(), 'logs/ep%03d-losses%.3f-val_loss%.3f.pth' % (
        (epoch + 1), total_loss / (epoch_step + 1), val_loss / (epoch_step_val + 1)))


def fit_one_epoch_no_val(model_train, model, loss_history, optimizer, epoch, epoch_step, gen, Epoch, cuda, dice_loss,
                         focal_loss, cls_weights, num_classes):
    total_loss = 0
    total_f_score = 0

    print('Start Train')
    with tqdm(total=epoch_step, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3) as pbar:
        for iteration, batch in enumerate(gen):
            if iteration >= epoch_step:
                break
            imgs, pngs, labels, depths = batch

            with torch.no_grad():
                imgs = torch.from_numpy(imgs).type(torch.FloatTensor)
                pngs = torch.from_numpy(pngs).long()
                labels = torch.from_numpy(labels).type(torch.FloatTensor)
                depths = torch.from_numpy(depths).type(torch.FloatTensor)
                depths = torch.unsqueeze(depths, 1)
                weights = torch.from_numpy(cls_weights)
                if cuda:
                    imgs = imgs.cuda()
                    pngs = pngs.cuda()
                    labels = labels.cuda()
                    depths = depths.cuda()
                    weights = weights.cuda()

            optimizer.zero_grad()

            sem_outputs, depth_outputs = model_train(imgs)

            # 语义分割部分的损失项计算
            if focal_loss:
                sem_loss = Focal_Loss(sem_outputs, pngs, weights, num_classes=num_classes)
            else:
                sem_loss = CE_Loss(sem_outputs, pngs, weights, num_classes=num_classes)

            if dice_loss:
                main_dice = Dice_loss(sem_outputs, labels)
                sem_loss = sem_loss + main_dice

            with torch.no_grad():
                # -------------------------------#
                #   计算f_score
                # -------------------------------#
                _f_score = f_score(sem_outputs, labels)

            # 深度值损失部分
            depth_loss = GRAD_LOSS(depths, depth_outputs)

            loss = sem_loss + depth_loss

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_f_score += _f_score.item()

            pbar.set_postfix(**{'total_loss': total_loss / (iteration + 1),
                                'f_score': total_f_score / (iteration + 1),
                                'lr': get_lr(optimizer)})
            pbar.update(1)

    print('Finish Train')

    loss_history.append_loss(total_loss / (epoch_step + 1))
    print('Finish Validation')
    print('Epoch:' + str(epoch + 1) + '/' + str(Epoch))
    print('Total Loss: %.3f' % (total_loss / (epoch_step + 1)))
    torch.save(model.state_dict(), 'logs/ep%03d-loss%.3f.pth' % ((epoch + 1), total_loss / (epoch_step + 1)))