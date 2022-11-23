import logging
import os
import time
import torch
import torch.nn as nn
from utils.meter import AverageMeter
from utils.metrics import R1_mAP_eval
from torch.cuda import amp
import torch.distributed as dist
from tqdm import tqdm

def do_train(cfg,
             model,
             class_net,
             class_net2,
             train_loader,
             val_loader,
             optimizer,
             scheduler,
             loss_fn,
             num_query, local_rank,class_optimizer,class2_optimizer,class_scheduler,
             class2_scheduler,target_train_loader,ide_creiteron,loss_classifier,t_loader):
    log_period = cfg.SOLVER.LOG_PERIOD
    eval_period = cfg.SOLVER.EVAL_PERIOD

    device = "cuda"
    epochs = cfg.SOLVER.MAX_EPOCHS

    logger = logging.getLogger("transreid.train")
    logger.info('start training')
    _LOCAL_PROCESS_GROUP = None
    if device:
        model.to(local_rank)
        class_net.to(local_rank)
        class_net2.to(local_rank)
        if torch.cuda.device_count() > 1 and cfg.MODEL.DIST_TRAIN:
            print('Using {} GPUs for training'.format(torch.cuda.device_count()))
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True)

    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    scaler = amp.GradScaler()
    # train

    # num of batch
    num_Batch = 100

    # self-train Num
    sl_train_num = 50

    # Num of source batch
    Num_Batch = 300

    # Print best result
    best_top1 = 0
    best_top5 = 0
    best_top10 = 0
    best_map = 0
    best_epoch = 0

    logger.info("The loss classifier para is C1_{},C2_{}, and Cunion_{}".format(loss_classifier[0], loss_classifier[1], loss_classifier[2]))

    for epoch in range(1, epochs + 1):
        start_time = time.time()
        loss_meter.reset()
        acc_meter.reset()
        evaluator.reset()
        scheduler.step(epoch)
        class_scheduler.step(epoch)
        class2_scheduler.step(epoch)
        model.train()
        # the epochs in self-train
        if epoch < sl_train_num:
            for i in range(Num_Batch):
                n_iter = i
                img, vid, target_cam, target_view = train_loader[0].next_one()
                optimizer.zero_grad()
                img = img.to(device)
                target = vid.to(device)
                target_cam = target_cam.to(device)
                target_view = target_view.to(device)
                with amp.autocast(enabled=True):
                    score, feat,_ = model(img, target, cam_label=target_cam, view_label=target_view )
                    loss = loss_fn(score, feat, target, target_cam)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                target_iters=t_loader[0]
                for i in range(len(target_iters)):
                    t_img, t_vid, t_target_cam, t_target_view = target_iters[i].next_one()
                    optimizer.zero_grad()
                    t_img = t_img.to(device)
                    t_target = t_vid.to(device)
                    with amp.autocast(enabled=True):
                        t_feat,t_score = model(t_img,modal=1,cam_label=t_target_cam)
                        t_loss = loss_fn(t_score, t_feat, t_target, t_target_cam)
                    scaler.scale(t_loss).backward()
                    scaler.step(optimizer)
                    scaler.update()

                if isinstance(score, list):
                    acc = (score[0].max(1)[1] == target).float().mean()
                else:
                    acc = (score.max(1)[1] == target).float().mean()

                loss_meter.update(loss.item(), img.shape[0])
                acc_meter.update(acc, 1)

                torch.cuda.synchronize()
                if (n_iter + 1) % log_period == 0:
                    logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}"
                                .format(epoch, (n_iter + 1), len(train_loader[1]),
                                        loss_meter.avg, acc_meter.avg, optimizer.state_dict()['param_groups'][0]['lr']))

            end_time = time.time()
            time_per_batch = (end_time - start_time) / (n_iter + 1)
            if cfg.MODEL.DIST_TRAIN:
                pass
            else:
                logger.info("Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]"
                        .format(epoch, time_per_batch, train_loader[1].batch_size / time_per_batch))

        # the epochs for classifier working
        else:
            class_net.train()
            class_net2.train()
            for i in range(num_Batch):
                n_iter = i
                train_iters = train_loader[0]
                target_iters=target_train_loader[0]
                img, vid, target_cam, target_view = train_iters.next_one()
                t_img, t_vid, t_target_cam, t_target_view = target_iters.next_one()

                optimizer.zero_grad()
                class_optimizer.zero_grad()
                class2_optimizer.zero_grad()
                img = img.to(device)
                target = vid.to(device)
                target_cam = target_cam.to(device)
                target_view = target_view.to(device)
                t_img = t_img.to(device)
                t_target = t_vid.to(device)
                t_target_cam = t_target_cam.to(device)
                t_target_view = t_target_view.to(device)

                if cfg.DATASETS.TARGET == 'dukemtmc':
                    Cam_Nero = [8, 9, 10]
                elif cfg.DATASETS.TARGET == 'market1501':
                    Cam_Nero = [6, 7, 8]
                elif cfg.DATASETS.TARGET == 'msmt17':
                    Cam_Nero = [15, 16, 17]

                # top -> head
                # m -> torso
                # b -> leg
                top_label = (torch.ones(t_target_cam.size()) * Cam_Nero[0]).int()
                m_label = (torch.ones(t_target_cam.size()) * Cam_Nero[1]).int()
                b_label = (torch.ones(t_target_cam.size()) * Cam_Nero[2]).int()

                top_label = top_label.to(device)
                m_label = m_label.to(device)
                b_label = b_label.to(device)

                with amp.autocast(enabled=True):
                    _ ,l_tfeat= model(t_img, t_target,modal=2, cam_label=t_target_cam, view_label=t_target_view)
                    t_top_feat,t_m_feat,t_b_feat = l_tfeat[0],l_tfeat[1],l_tfeat[2]
                    t_top_cls,t_top_cls2,_ = class_net(t_top_feat.detach())
                    t_m_cls,t_m_cls2,_ = class_net(t_m_feat.detach())
                    t_b_cls,t_b_cls2,_ = class_net(t_b_feat.detach())

                    loss_t = loss_classifier[0]*(ide_creiteron(t_top_cls, t_target_cam)+ide_creiteron(t_m_cls, t_target_cam)+ide_creiteron(t_b_cls, t_target_cam))\
                             +loss_classifier[1]*(ide_creiteron(t_top_cls2, t_target_cam)+ide_creiteron(t_m_cls2, t_target_cam)+ide_creiteron(t_b_cls2, t_target_cam))

                    _, _,l_feat = model(img, target, cam_label=target_cam, view_label=target_view)
                    top_feat,m_feat,b_feat = l_feat[0],l_feat[1],l_feat[2]
                    top_cls,top_cls2,_= class_net(top_feat.detach())
                    m_cls,m_cls2,_ = class_net(m_feat.detach())
                    b_cls,b_cls2,_ = class_net(b_feat.detach())

                    try:
                        loss_body = loss_classifier[0]*(ide_creiteron(top_cls, top_label)+ide_creiteron(m_cls, m_label)+ide_creiteron(b_cls, b_label))\
                                   +loss_classifier[1]*(ide_creiteron(top_cls2, top_label)+ide_creiteron( m_cls2, m_label)+ide_creiteron(b_cls2, b_label))
                    except:
                        print('Error in loss_body')

                    loss = loss_t + loss_body

                scaler.scale(loss).backward()
                scaler.step(class_optimizer)
                scaler.update()

                # target
                with amp.autocast(enabled=True):
                    score, feat,l_tfeat = model(img, target, cam_label=target_cam, view_label=target_view)
                    _,l_tfeat = model(t_img, t_target,modal=2, cam_label=target_cam, view_label=target_view )

                    id_loss = loss_fn(score, feat, target, target_cam)
                    t_top_feat,t_m_feat,t_b_feat = l_tfeat[0],l_tfeat[1],l_tfeat[2]
                    _,_,t_top_clss = class_net(t_top_feat)
                    _,_,t_m_clss = class_net(t_m_feat)
                    _,_,t_b_clss = class_net(t_b_feat)

                    loss = loss_classifier[2]*(ide_creiteron(t_top_clss, top_label) + ide_creiteron(t_m_clss, m_label) + ide_creiteron(t_b_clss, b_label))\
                            +id_loss

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                target_iters=t_loader[0]
                for i in range(len(target_iters)):
                    t_img, t_vid, t_target_cam, t_target_view = target_iters[i].next_one()
                    optimizer.zero_grad()
                    t_img = t_img.to(device)
                    t_target = t_vid.to(device)
                    with amp.autocast(enabled=True):
                        t_feat,t_score = model(t_img,modal=1,cam_label=t_target_cam)
                        t_loss = loss_fn(t_score, t_feat, t_target, t_target_cam)
                    scaler.scale(t_loss).backward()
                    scaler.step(optimizer)
                    scaler.update()

                if isinstance(score, list):
                    acc = (score[0].max(1)[1] == target).float().mean()
                else:
                    acc = (score.max(1)[1] == target).float().mean()

                loss_meter.update(loss.item(), img.shape[0])
                acc_meter.update(acc, 1)

                torch.cuda.synchronize()

                if (n_iter + 1) % log_period == 0:
                    logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}"
                                .format(epoch, (n_iter + 1), len(train_loader),
                                        loss_meter.avg, acc_meter.avg, optimizer.state_dict()['param_groups'][0]['lr']))

            end_time = time.time()
            time_per_batch = (end_time - start_time) / (n_iter + 1)

            if cfg.MODEL.DIST_TRAIN:
                pass
            else:
                logger.info("Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]"
                            .format(epoch, time_per_batch, train_loader[1].batch_size / time_per_batch))

        if epoch >= sl_train_num:
            if epoch % eval_period == 0:
                # # If train with multi-gpu ddp mode, options: 'True', 'False'
                # not use in windows
                if cfg.MODEL.DIST_TRAIN:
                    if dist.get_rank() == 0:
                        model.eval()
                        for n_iter, (img, vid, camid, camids, target_view, _) in enumerate(val_loader):
                            with torch.no_grad():
                                img = img.to(device)
                                camids = camids.to(device)
                                target_view = target_view.to(device)
                                feat = model(img, cam_label=camids, view_label=target_view)
                                evaluator.update((feat, vid, camid))
                        cmc, mAP, _, _, _, _, _ = evaluator.compute()
                        logger.info("Validation Results - Epoch: {}".format(epoch))
                        logger.info("mAP: {:.1%}".format(mAP))
                        for r in [1, 5, 10]:
                            logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))

                        if cmc[0] > best_top1:
                            best_top1 = max(cmc[0], best_top1)
                            best_top5 = cmc[4]
                            best_top10 = cmc[9]
                            best_map = mAP
                            best_epoch = epoch
                            # save the model
                            # torch.save(model.state_dict(),
                            #            os.path.join(cfg.OUTPUT_DIR, 'bestResult_{}.pth'.format(best_epoch)))
                        torch.cuda.empty_cache()
                else:
                    model.eval()
                    for n_iter, (img, vid, camid, camids, target_view, _) in enumerate(val_loader):
                        with torch.no_grad():
                            img = img.to(device)
                            camids = camids.to(device)
                            target_view = target_view.to(device)
                            feat = model(img, cam_label=camids, view_label=target_view)
                            evaluator.update((feat, vid, camid))
                    cmc, mAP, _, _, _, _, _ = evaluator.compute()
                    logger.info("Validation Results - Epoch: {}".format(epoch))
                    logger.info("mAP: {:.1%}".format(mAP))
                    for r in [1, 5, 10]:
                        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))

                    # save the best result
                    if cmc[0] > best_top1:
                        best_top1 = max(cmc[0], best_top1)
                        best_top5 = cmc[4]
                        best_top10 = cmc[9]
                        best_map = mAP
                        best_epoch = epoch
                        # save the model
                        torch.save(model.state_dict(),
                                   os.path.join(cfg.OUTPUT_DIR, 'bestResult_{}.pth'.format(best_epoch)))

                    logger.info("Best Results - Epoch: {}".format(best_epoch))
                    logger.info("Best Rank-1  :{:.1%} and mAP: {:.1%}".format(best_top1, best_map))

                    torch.cuda.empty_cache()
    # show the best result
    logger.info("Best Results - Epoch: {}".format(best_epoch))
    logger.info("mAP: {:.1%}".format(best_map))
    logger.info("CMC curve, Rank-1  :{:.1%}".format(best_top1))
    logger.info("CMC curve, Rank-5  :{:.1%}".format(best_top5))
    logger.info("CMC curve, Rank-10 :{:.1%}".format(best_top10))

def do_inference(cfg,
                 model,
                 val_loader,
                 num_query):
    device = "cuda"
    logger = logging.getLogger("transreid.test")
    logger.info("Enter inferencing")

    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)

    evaluator.reset()

    if device:
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for inference'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model)
        model.to(device)

    # refresh model
    model.eval()
    img_path_list = []

    for n_iter, (img, pid, camid, camids, target_view, imgpath) in enumerate(val_loader):
        with torch.no_grad():
            img = img.to(device)
            camids = camids.to(device)
            target_view = target_view.to(device)
            feat = model(img, cam_label=camids, view_label=target_view)
            evaluator.update((feat, pid, camid))
            img_path_list.extend(imgpath)

    cmc, mAP, _, _, _, _, _ = evaluator.compute()
    logger.info("Validation Results ")
    logger.info("mAP: {:.1%}".format(mAP))
    for r in [1, 5, 10]:
        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
    return cmc[0], cmc[4]




def mean_weight(model):
    # softmax = nn.Softmax(dim=0)
    w1_0 = model.classifier[0].weight
    w2_0 = model.classifier2[0].weight
    w1_3 = model.classifier[3].weight
    w2_3 = model.classifier2[3].weight
    w1_6 = model.classifier[6].weight
    w2_6 = model.classifier2[6].weight
    w1_9 = model.classifier[9].weight
    w2_9 = model.classifier2[9].weight


    loss_weight = weight_diff(w1_0, w2_0) + weight_diff(w1_3, w2_3) + \
                  weight_diff(w1_6, w2_6) + weight_diff(w1_9, w2_9)
    loss_weight = loss_weight/4

    return loss_weight


def weight_diff(w1,w2):
    w2 = w2.view(-1)
    w1 = w1.view(-1)
    loss = (torch.matmul(w1,w2)/(torch.norm(w1)*torch.norm(w2))+1)
    return loss