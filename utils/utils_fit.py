import torch
import os
# from losses.semantic_losses import CE_Loss, Dice_loss, Focal_Loss
from losses.depth_losses import GRAD_LOSS, BerHu_Loss
from tqdm import tqdm

from segmentation_models_pytorch.losses import SoftBCEWithLogitsLoss, DiceLoss, FocalLoss

from utils.utils import get_lr
from utils.utils_metrics import f_score, binary_mean_iou

bce_loss_fn = SoftBCEWithLogitsLoss()
focal_loss_fn = FocalLoss(mode='binary')
dice_loss_fn = DiceLoss(mode='binary', from_logits=True)


def fit_one_epoch(model_train, loss_history, optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val, Epoch,
                  cuda, dice_loss, focal_loss, depth_loss_factor):
    train_loss = 0
    val_loss = 0

    f_score = 0
    sem_loss_value = 0
    depth_loss_value = 0
    miou = 0

    model_train.train()
    stage = 'train'
    print('Start Train')
    with tqdm(total=epoch_step, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3) as pbar:
        for iteration, batch in enumerate(gen):
            if iteration >= epoch_step:
                break

            optimizer.zero_grad()
            step_output = fit_one_step(batch, model_train, cuda, dice_loss,
                                       focal_loss, depth_loss_factor, stage)

            loss = step_output['loss']

            loss.backward()
            optimizer.step()

            sem_loss_value += step_output['sem_loss']
            depth_loss_value += step_output['depth_loss']
            miou += step_output['miou']

            train_loss += loss.item()
            f_score += step_output['f_score']

            log_dict = {'total_loss': train_loss / (iteration + 1),
                        'f_score': f_score / (iteration + 1),
                        'sem_loss': sem_loss_value / (iteration + 1),
                        'depth_loss': depth_loss_value / (iteration + 1),
                        'miou': miou / (iteration + 1),
                        'lr': get_lr(optimizer)}

            loss_history.log_dict(log_dict, stage)

            pbar.set_postfix(**log_dict)
            pbar.update(1)

    print('Finish Train')

    stage = 'val'
    f_score = 0
    sem_loss_value = 0
    depth_loss_value = 0
    miou = 0

    model_train.eval()
    print('Start Validation')
    with tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3) as pbar:
        for iteration, batch in enumerate(gen_val):
            if iteration >= epoch_step_val:
                break

            with torch.no_grad():
                fit_one_step(batch, model_train, cuda, dice_loss,
                             focal_loss, depth_loss_factor, stage)

                sem_loss_value += step_output['sem_loss']
                depth_loss_value += step_output['depth_loss']
                miou += step_output['miou']

                val_loss += loss.item()
                f_score += step_output['f_score']

                log_dict = {'total_loss': val_loss / (iteration + 1),
                            'f_score': f_score / (iteration + 1),
                            'sem_loss': sem_loss_value / (iteration + 1),
                            'depth_loss': depth_loss_value / (iteration + 1),
                            'miou': miou / (iteration + 1),
                            'lr': get_lr(optimizer)}

            loss_history.log_dict(log_dict, stage)

            pbar.set_postfix(**log_dict)
            pbar.update(1)

    epoch_train_loss = train_loss / (epoch_step + 1)
    epoch_val_loss = val_loss / (epoch_step_val + 1)
    epoch_miou = miou / (epoch_step_val + 1)

    loss_history.append_loss(epoch_train_loss, epoch_val_loss)
    print('Finish Validation')

    print('Epoch:' + str(epoch + 1) + '/' + str(Epoch))
    print('Total Loss: %.3f || Val Loss: %.3f ' % (epoch_train_loss, epoch_val_loss))

    return epoch_miou


def fit_one_step(batch, model, cuda, dice_loss,
                 focal_loss, depth_loss_factor, stage):
    imgs, pngs, labels, depths = batch

    with torch.no_grad():
        if cuda:
            imgs = imgs.cuda()
            pngs = pngs.cuda()
            labels = labels.cuda()
            depths = depths.cuda()

    outputs = model(imgs)

    outputs = list(outputs) if isinstance(outputs, tuple) else [outputs]

    sem_outputs = outputs[0]

    # 语义分割部分的损失项计算
    if focal_loss:
        sem_loss = focal_loss_fn(sem_outputs, pngs.float())
    else:
        sem_loss = bce_loss_fn(sem_outputs, pngs.float())

    if dice_loss:
        main_dice = dice_loss_fn(sem_outputs, pngs)
        sem_loss = sem_loss + main_dice

    sem_loss_value = sem_loss.item()

    with torch.no_grad():
        # -------------------------------#
        #   计算f_score
        # -------------------------------#
        _f_score = f_score(sem_outputs, labels)
        _miou = binary_mean_iou(sem_outputs, labels)

    # 深度值损失部分
    if len(outputs) > 1:
        depth_outputs = outputs[1]
        d1_loss = BerHu_Loss(depths, depth_outputs)
        # d2_loss = GRAD_LOSS(depths, depth_outputs)

        depth_loss = d1_loss  # + d2_loss
        depth_loss_value = depth_loss.item()

    else:
        depth_loss = 0
        depth_loss_value = 0

    loss = sem_loss + depth_loss_factor * depth_loss

    return {'loss': loss,
            'sem_loss': sem_loss_value,
            'depth_loss': depth_loss_value,
            'f_score': _f_score.item(),
            'miou': _miou.item()}


# 原来的fit_one_epoch函数
'''
# 原来的fit_one_epoch函数
# def fit_one_epoch_old(model_train, model, loss_history, optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val,
#                       Epoch,
#                       cuda, dice_loss, focal_loss, cls_weights, depth_loss_factor):
#     total_loss = 0
#     total_f_score = 0
#
#     val_loss = 0
#     val_f_score = 0
#
#     model_train.train()
#     print('Start Train')
#     with tqdm(total=epoch_step, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3) as pbar:
#         for iteration, batch in enumerate(gen):
#             if iteration >= epoch_step:
#                 break
#             imgs, pngs, labels, depths = batch
#
#             with torch.no_grad():
#                 imgs = imgs.cuda()
#                 pngs = pngs.cuda()
#                 labels = labels.cuda()
#                 depths = depths.cuda()
#
#             optimizer.zero_grad()
#
#             outputs = model_train(imgs)
#
#             outputs = list(outputs) if isinstance(outputs, tuple) else [outputs]
#
#             sem_outputs = outputs[0]
#
#             # 语义分割部分的损失项计算
#             if focal_loss:
#                 sem_loss = focal_loss_fn(sem_outputs, pngs.float())
#             else:
#                 sem_loss = bce_loss_fn(sem_outputs, pngs.float())
#
#             if dice_loss:
#                 main_dice = dice_loss_fn(sem_outputs, pngs)
#                 sem_loss = sem_loss + main_dice
#
#             with torch.no_grad():
#                 # -------------------------------#
#                 #   计算f_score
#                 # -------------------------------#
#                 _f_score = f_score(sem_outputs, pngs)
#                 miou = binary_mean_iou(sem_outputs, pngs)
#
#             # 深度值损失部分
#             if len(outputs) > 1:
#                 depth_outputs = outputs[1]
#                 d1_loss = BerHu_Loss(depths, depth_outputs)
#                 # d2_loss = GRAD_LOSS(depths, depth_outputs)
#
#                 depth_loss = d1_loss  # + d2_loss
#                 depth_loss_value = depth_loss.item()
#
#             else:
#                 depth_loss = 0
#                 depth_loss_value = 0
#
#             loss = sem_loss + depth_loss_factor * depth_loss
#
#             loss.backward()
#             optimizer.step()
#
#             total_loss += loss.item()
#             total_f_score += _f_score.item()
#
#             pbar.set_postfix(**{'loss': total_loss / (iteration + 1),
#                                 'f_score': total_f_score / (iteration + 1),
#                                 'sem_loss': sem_loss.item(),
#                                 'depth_loss': depth_loss_value,
#                                 'lr': get_lr(optimizer)})
#             pbar.update(1)
#
#     print('Finish Train')
#
#     model_train.eval()
#     print('Start Validation')
#     with tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3) as pbar:
#         for iteration, batch in enumerate(gen_val):
#             if iteration >= epoch_step_val:
#                 break
#             imgs, pngs, labels, depths = batch
#             with torch.no_grad():
#                 # imgs = torch.from_numpy(imgs).type(torch.FloatTensor)
#                 # pngs = torch.from_numpy(pngs).long()
#                 # labels = torch.from_numpy(labels).type(torch.FloatTensor)
#                 # depths = torch.from_numpy(depths).type(torch.FloatTensor)
#                 weights = torch.from_numpy(cls_weights)
#                 if cuda:
#                     imgs = imgs.cuda()
#                     pngs = pngs.cuda()
#                     labels = labels.cuda()
#                     depths = depths.cuda()
#                     weights = weights.cuda()
#
#                 outputs = model_train(imgs)
#
#                 outputs = list(outputs) if isinstance(outputs, tuple) else [outputs]
#
#                 sem_outputs = outputs[0]
#
#                 # 语义分割部分的损失项计算
#                 if focal_loss:
#                     sem_loss = focal_loss_fn(sem_outputs, pngs.float())
#                 else:
#                     sem_loss = bce_loss_fn(sem_outputs, pngs.float())
#
#                 if dice_loss:
#                     main_dice = dice_loss_fn(sem_outputs, labels)
#                     sem_loss = sem_loss + main_dice
#
#                 with torch.no_grad():
#                     # -------------------------------#
#                     #   计算f_score
#                     # -------------------------------#
#                     _f_score = f_score(sem_outputs, labels)
#
#                 # 深度值损失部分
#                 if len(outputs) > 1:
#                     depth_outputs = outputs[1]
#                     d1_loss = BerHu_Loss(depths, depth_outputs)
#                     # d2_loss = GRAD_LOSS(depths, depth_outputs)
#
#                     depth_loss = d1_loss  # + d2_loss
#                     depth_loss_value = depth_loss.item()
#
#                 else:
#                     depth_loss = 0
#                     ddpth_loss_value = 0
#
#                 loss = sem_loss + depth_loss_factor * depth_loss
#
#                 val_loss += loss.item()
#                 val_f_score += _f_score.item()
#
#             pbar.set_postfix(**{'total_loss': val_loss / (iteration + 1),
#                                 'f_score': val_f_score / (iteration + 1),
#                                 'sem_loss': sem_loss.item(),
#                                 'depth_loss': depth_loss_value,
#                                 'lr': get_lr(optimizer)})
#             pbar.update(1)
#
#     loss_history.append_loss(total_loss / (epoch_step + 1), val_loss / (epoch_step_val + 1))
#     print('Finish Validation')
#     print('Epoch:' + str(epoch + 1) + '/' + str(Epoch))
#     print('Total Loss: %.3f || Val Loss: %.3f ' % (total_loss / (epoch_step + 1), val_loss / (epoch_step_val + 1)))
#
#     # SAVE MODEL
#     if not (epoch + 1) % 10:
#         torch.save(model.state_dict(), os.path.join(loss_history.save_path, 'ep%03d.pth' % (epoch + 1)))
'''
