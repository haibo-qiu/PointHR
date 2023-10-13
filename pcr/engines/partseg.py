import os
import sys
import time
import shutil
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
import torch.distributed as dist
import pointops
from torch.nn.parallel import DistributedDataParallel
from functools import partial
from tensorboardX import SummaryWriter
from collections import Counter

import pcr.utils.comm as comm
from pcr.datasets import build_dataset, point_collate_fn, collate_fn
from pcr.models import build_model
from pcr.utils.logger import get_root_logger
from pcr.utils.optimizer import build_optimizer
from pcr.utils.scheduler import build_scheduler
from pcr.utils.losses import build_criteria
from pcr.utils.events import EventStorage
from pcr.utils.misc import intersection_and_union_gpu
from pcr.utils.env import get_random_seed, set_seed
from pcr.utils.config import Config, DictAction


def torch_grouping_operation(features, idx):
    r"""from torch points kernels
    Parameters
    ----------
    features : torch.Tensor
        (B, C, N) tensor of features to group
    idx : torch.Tensor
        (B, npoint, nsample) tensor containing the indicies of features to group with

    Returns
    -------
    torch.Tensor
        (B, C, npoint, nsample) tensor
    """
    all_idx = idx.reshape(idx.shape[0], -1)
    all_idx = all_idx.unsqueeze(1).repeat(1, features.shape[1], 1)
    grouped_features = features.gather(2, all_idx)
    return grouped_features.reshape(idx.shape[0], features.shape[1], idx.shape[1], idx.shape[2])

@torch.no_grad()
def knn_point(k, query, support=None):
    """Get the distances and indices to a fixed number of neighbors
        Args:
            support ([tensor]): [B, N, C]
            query ([tensor]): [B, M, C]

        Returns:
            [int]: neighbor idx. [B, M, K]
    """
    if support is None:
        support = query
    dist = torch.cdist(query, support)
    k_dist = dist.topk(k=k, dim=-1, largest=False, sorted=True)
    return k_dist.values, k_dist.indices

def batched_bincount(x, dim, max_value):
    target = torch.zeros(x.shape[0], max_value, dtype=x.dtype, device=x.device)
    values = torch.ones_like(x)
    target.scatter_add_(dim, x, values)
    return target


def part_seg_refinement(pred, pos, cls, cls2parts, n=10):
    pred_np = pred.cpu().data.numpy()
    for shape_idx in range(pred.size(0)):  # sample_idx
        parts = cls2parts[cls[shape_idx]]
        counter_part = Counter(pred_np[shape_idx])
        if len(counter_part) > 1:
            for i in counter_part:
                if counter_part[i] < n or i not in parts:
                    less_idx = np.where(pred_np[shape_idx] == i)[0]
                    less_pos = pos[shape_idx][less_idx]
                    knn_idx = knn_point(n + 1, torch.unsqueeze(less_pos, axis=0),
                                        torch.unsqueeze(pos[shape_idx], axis=0))[1]
                    neighbor = torch_grouping_operation(pred[shape_idx:shape_idx + 1].unsqueeze(1), knn_idx)[0][0]
                    counts = batched_bincount(neighbor, 1, cls2parts[-1][-1] + 1)
                    counts[:, i] = 0
                    pred[shape_idx][less_idx] = counts.max(dim=1)[1]
    return pred


def get_ins_mious(pred, target, cls, cls2parts,
                  multihead=False,
                  ):
    """Get the Shape IoU
    shape IoU: the mean part iou for each shape
    Args:
        pred (_type_): _description_
        target (_type_): _description_
        num_classes (_type_): _description_
    Returns:
        _type_: _description_
    """
    ins_mious = []
    for shape_idx in range(pred.shape[0]):  # sample_idx
        part_ious = []
        parts = cls2parts[cls[shape_idx]]
        if multihead:
            parts = np.arange(len(parts))

        for part in parts:
            pred_part = pred[shape_idx] == part
            target_part = target[shape_idx] == part
            I = torch.logical_and(pred_part, target_part).sum()
            U = torch.logical_or(pred_part, target_part).sum()
            if U == 0:
                iou = torch.tensor(100, device=pred.device, dtype=torch.float32)
            else:
                iou = I * 100 / float(U)
            part_ious.append(iou)
        ins_mious.append(torch.mean(torch.stack(part_ious)))
    return ins_mious

def create_ddp_model(model, *, fp16_compression=False, **kwargs):
    """
    Create a DistributedDataParallel model if there are >1 processes.
    Args:
        model: a torch.nn.Module
        fp16_compression: add fp16 compression hooks to the ddp object.
            See more at https://pytorch.org/docs/stable/ddp_comm_hooks.html#torch.distributed.algorithms.ddp_comm_hooks.default_hooks.fp16_compress_hook
        kwargs: other arguments of :module:`torch.nn.parallel.DistributedDataParallel`.
    """
    if comm.get_world_size() == 1:
        return model
    # kwargs['find_unused_parameters'] = True
    if "device_ids" not in kwargs:
        kwargs["device_ids"] = [comm.get_local_rank()]
        if "output_device" not in kwargs:
            kwargs["output_device"] = [comm.get_local_rank()]
    ddp = DistributedDataParallel(model,  **kwargs)
    if fp16_compression:
        from torch.distributed.algorithms.ddp_comm_hooks import default as comm_hooks

        ddp.register_comm_hook(state=None, hook=comm_hooks.fp16_compress_hook)
    return ddp


def worker_init_fn(worker_id, num_workers, rank, seed):
    """Worker init func for dataloader.

    The seed of each worker equals to num_worker * rank + worker_id + user_seed

    Args:
        worker_id (int): Worker id.
        num_workers (int): Number of workers.
        rank (int): The rank of current process.
        seed (int): The random seed to use.
    """

    worker_seed = num_workers * rank + worker_id + seed
    set_seed(worker_seed)


class Trainer:
    def __init__(self, cfg):
        self.epoch = 0
        self.start_epoch = 0
        self.max_epoch = cfg.eval_epoch
        # TODO: add to hook
        # BestCheckpointer
        self.eval_metric = cfg.eval_metric
        self.best_metric_value = -torch.inf
        # TimeEstimator
        self.iter_end_time = None
        self.max_iter = None

        self.logger = get_root_logger(log_file=os.path.join(cfg.save_path, "train.log"),
                                      file_mode='a' if cfg.resume else 'w')
        self.logger.info("=> Loading config ...")
        self.cfg = cfg
        self.logger.info(f"Save path: {cfg.save_path}")
        self.logger.info(f"Config:\n{cfg.pretty_text}")
        self.storage: EventStorage
        self.logger.info("=> Building model ...")
        self.model = self.build_model()
        self.logger.info("=> Building writer ...")
        self.writer = self.build_writer()
        self.logger.info("=> Building train dataset & dataloader ...")
        self.train_loader = self.build_train_loader()
        self.logger.info("=> Building val dataset & dataloader ...")
        self.val_loader = self.build_val_loader()
        self.logger.info("=> Building criteria, optimize, scheduler, scaler(amp) ...")
        self.criteria = self.build_criteria()
        self.optimizer = self.build_optimizer()
        self.scheduler = self.build_scheduler()
        self.scaler = self.build_scaler()
        self.logger.info("=> Checking load & resume ...")
        self.resume_or_load()

    def train(self):
        with EventStorage() as self.storage:
            # => before train
            self.logger.info('>>>>>>>>>>>>>>>> Start Training >>>>>>>>>>>>>>>>')
            self.max_iter = self.max_epoch * len(self.train_loader)
            for self.epoch in range(self.start_epoch, self.max_epoch):
                # => before epoch
                if comm.get_world_size() > 1:
                    # fix epoch shuffle pattern make training better
                    self.train_loader.sampler.set_epoch(self.start_epoch)
                self.model.train()
                self.iter_end_time = time.time()
                # => run_epoch
                for i, input_dict in enumerate(self.train_loader):
                    # => before_step
                    # => run_step
                    self.run_step(i, input_dict)
                    # => after_step
                # => after epoch
                self.after_epoch()
            # => after train
            self.logger.info('==>Training done!\nBest {}: {:.4f}'.format(
                self.cfg.eval_metric, self.best_metric_value))
            if self.writer is not None:
                self.writer.close()


    def run_step(self, i, input_dict):
        data_time = time.time() - self.iter_end_time
        for key in input_dict.keys():
            input_dict[key] = input_dict[key].cuda(non_blocking=True)

        with torch.cuda.amp.autocast(enabled=self.cfg.enable_amp):
            output = self.model(input_dict)
            loss = self.criteria(output, input_dict["label"].long())
        self.optimizer.zero_grad()
        if self.cfg.enable_amp:
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            self.optimizer.step()
        self.scheduler.step()
        if self.cfg.empty_cache:
            torch.cuda.empty_cache()
        n = input_dict["coord"].size(0)
        if comm.get_world_size() > 1:
            loss *= n
            count = input_dict["label"].new_tensor([n], dtype=torch.long)
            dist.all_reduce(loss), dist.all_reduce(count)
            n = count.item()
            loss /= n
        # intersection, union, target = \
        #     intersection_and_union_gpu(
        #         output.max(1)[1], input_dict["label"], self.cfg.data.num_classes, self.cfg.data.ignore_label)
        # if comm.get_world_size() > 1:
        #     dist.all_reduce(intersection), dist.all_reduce(union), dist.all_reduce(target)
        # intersection, union, target = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy()
        # Here there is no need to sync since sync happened in dist.all_reduce
        batch_time = time.time() - self.iter_end_time
        self.iter_end_time = time.time()

        # self.storage.put_scalar("intersection", intersection)
        # self.storage.put_scalar("union", union)
        # self.storage.put_scalar("target", target)
        self.storage.put_scalar("loss", loss.item(), n=n)
        self.storage.put_scalar("data_time", data_time)
        self.storage.put_scalar("batch_time", batch_time)

        # calculate remain time
        current_iter = self.epoch * len(self.train_loader) + i + 1
        remain_iter = self.max_iter - current_iter
        remain_time = remain_iter * self.storage.history("batch_time").avg
        t_m, t_s = divmod(remain_time, 60)
        t_h, t_m = divmod(t_m, 60)
        remain_time = '{:02d}:{:02d}:{:02d}'.format(int(t_h), int(t_m), int(t_s))
        self.logger.info('Train: [{epoch}/{max_epoch}][{iter}/{max_iter}] '
                         'Scan {batch_size} ({points_num}) '
                         'Data {data_time_val:.3f} ({data_time_avg:.3f}) '
                         'Batch {batch_time_val:.3f} ({batch_time_avg:.3f}) '
                         'Remain {remain_time} '
                         'Lr {lr:.4f} '
                         'Loss {loss:.4f} '.format(epoch=self.epoch + 1, max_epoch=self.max_epoch, iter=i + 1,
                                                   max_iter=len(self.train_loader),
                                                   batch_size=len(input_dict["offset"]),
                                                   points_num=input_dict["offset"][-1],
                                                   data_time_val=data_time,
                                                   data_time_avg=self.storage.history("data_time").avg,
                                                   batch_time_val=batch_time,
                                                   batch_time_avg=self.storage.history("batch_time").avg,
                                                   remain_time=remain_time,
                                                   lr=self.optimizer.state_dict()['param_groups'][0]['lr'],
                                                   loss=loss.item()))
        if i == 0:
            # drop data_time and batch_time for the first iter
            self.storage.history("data_time").reset()
            self.storage.history("batch_time").reset()
        if self.writer is not None:
            self.writer.add_scalar('lr', self.optimizer.state_dict()['param_groups'][0]['lr'], current_iter)
            self.writer.add_scalar('train_batch/loss', loss.item(), current_iter)
            # self.writer.add_scalar('train_batch/mIoU', np.mean(intersection / (union + 1e-10)), current_iter)
            # self.writer.add_scalar('train_batch/mAcc', np.mean(intersection / (target + 1e-10)), current_iter)
            # self.writer.add_scalar('train_batch/allAcc', np.sum(intersection) / (np.sum(target) + 1e-10), current_iter)

    def after_epoch(self):
        loss_avg = self.storage.history("loss").avg
        # intersection = self.storage.history("intersection").total
        # union = self.storage.history("union").total
        # target = self.storage.history("target").total
        # iou_class = intersection / (union + 1e-10)
        # acc_class = intersection / (target + 1e-10)
        # m_iou = np.mean(iou_class)
        # m_acc = np.mean(acc_class)
        # all_acc = sum(intersection) / (sum(target) + 1e-10)
        # self.logger.info('Train result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(
        #     m_iou, m_acc, all_acc))
        self.logger.info('Train result: loss {:.4f}.'.format(loss_avg))
        current_epoch = self.epoch + 1
        if self.writer is not None:
            self.writer.add_scalar('train/loss', loss_avg, current_epoch)
            # self.writer.add_scalar('train/mIoU', m_iou, current_epoch)
            # self.writer.add_scalar('train/mAcc', m_acc, current_epoch)
            # self.writer.add_scalar('train/allAcc', all_acc, current_epoch)
        self.storage.reset_histories()
        if self.cfg.evaluate:
            self.eval()
        self.save_checkpoint()
        self.storage.reset_histories()

    def eval(self):
        self.logger.info('>>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>')
        cls_mious = torch.zeros(len(self.cls2parts), dtype=torch.float32).cuda(non_blocking=True)
        cls_nums = torch.zeros(len(self.cls2parts), dtype=torch.int32).cuda(non_blocking=True)
        ins_miou_list = []
        self.model.eval()
        self.iter_end_time = time.time()
        for i, input_dict in enumerate(self.val_loader):
            batch_size = len(input_dict['offset'])
            data_time = time.time() - self.iter_end_time
            for key in input_dict.keys():
                input_dict[key] = input_dict[key].cuda(non_blocking=True)
            with torch.no_grad():
                output = self.model(input_dict)
            loss = self.criteria(output, input_dict["label"].long())
            n = input_dict["coord"].size(0)
            if comm.get_world_size() > 1:
                loss *= n
                count = input_dict["label"].new_tensor([n], dtype=torch.long)
                dist.all_reduce(loss), dist.all_reduce(count)
                n = count.item()
                loss /= n
            # TODO: add to evaluator
            pred = output.max(1)[1]
            label = input_dict["label"]

            pred = pred.reshape(batch_size, -1)
            label = label.reshape(batch_size, -1)
            coord = input_dict["coord"].reshape(batch_size, -1, input_dict["coord"].shape[-1])

            if "origin_coord" in input_dict.keys():
                idx, _ = pointops.knn_query(1, input_dict["coord"].float(), input_dict["offset"].int(),
                                            input_dict["origin_coord"].float(), input_dict["origin_offset"].int())

                pred = pred[idx.flatten().long()]
                label = input_dict["origin_label"]

            part_seg_refinement(pred, coord, input_dict["cls_token"], self.cls2parts)
            batch_ins_mious = get_ins_mious(pred, label, input_dict["cls_token"], self.cls2parts)
            ins_miou_list += batch_ins_mious
            # per category iou at each batch_size:
            for shape_idx in range(batch_size):  # sample_idx
                cur_gt_label = int(input_dict["cls_token"][shape_idx].cpu().numpy())
                # add the iou belongs to this cat
                cls_mious[cur_gt_label] += batch_ins_mious[shape_idx]
                cls_nums[cur_gt_label] += 1

            # Here there is no need to sync since sync happened in dist.all_reduce
            batch_time = time.time() - self.iter_end_time
            self.iter_end_time = time.time()

            self.storage.put_scalar("loss", loss.item(), n=n)
            self.storage.put_scalar("data_time", data_time)
            self.storage.put_scalar("batch_time", batch_time)
            self.logger.info('Test: [{iter}/{max_iter}] '
                             'Data {data_time_val:.3f} ({data_time_avg:.3f}) '
                             'Batch {batch_time_val:.3f} ({batch_time_avg:.3f}) '
                             'Loss {loss:.4f} '.format(iter=i + 1,
                                                       max_iter=len(self.val_loader),
                                                       data_time_val=data_time,
                                                       data_time_avg=self.storage.history("data_time").avg,
                                                       batch_time_val=batch_time,
                                                       batch_time_avg=self.storage.history("batch_time").avg,
                                                       loss=loss.item()))

        ins_mious_sum, count = torch.sum(torch.stack(ins_miou_list)), torch.tensor(len(ins_miou_list)).cuda()
        if comm.get_world_size() > 1:
            dist.all_reduce(cls_mious), dist.all_reduce(cls_nums), dist.all_reduce(ins_mious_sum), dist.all_reduce(count)

        for cat_idx in range(len(self.cls2parts)):
            # indicating this cat is included during previous iou appending
            if cls_nums[cat_idx] > 0:
                cls_mious[cat_idx] = cls_mious[cat_idx] / cls_nums[cat_idx]

        ins_miou = ins_mious_sum/count
        cls_miou = torch.mean(cls_mious)

        loss_avg = self.storage.history("loss").avg
        self.storage.put_scalar("Ins_mIoU", ins_miou)
        self.storage.put_scalar("Cls_mIoU", cls_miou)
        self.logger.info('Val result: Ins_mIoU/Cls_mIoU {:.4f}/{:.4f}.'.format(
            ins_miou, cls_miou))
        for i in range(len(self.cls2parts)):
            self.logger.info('Class_{idx}-{name} Result: iou {iou:.4f}'.format(
                idx=i, name=self.categories[i], iou=cls_mious[i]))
        current_epoch = self.epoch + 1
        if self.writer is not None:
            self.writer.add_scalar('val/loss', loss_avg, current_epoch)
            self.writer.add_scalar('val/mIoU', ins_miou, current_epoch)
        self.logger.info('<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<')

    def save_checkpoint(self):
        if comm.is_main_process():
            is_best = False
            current_metric_value = self.storage.latest()[self.cfg.eval_metric][0] if self.cfg.evaluate else 0
            if self.cfg.evaluate and current_metric_value > self.best_metric_value:
                self.best_metric_value = current_metric_value
                is_best = True

            filename = os.path.join(self.cfg.save_path, 'model', 'model_last.pth')
            self.logger.info('Saving checkpoint to: ' + filename)
            torch.save({'epoch': self.epoch + 1,
                        'state_dict': self.model.state_dict(),
                        'optimizer': self.optimizer.state_dict(),
                        'scheduler': self.scheduler.state_dict(),
                        'scaler': self.scaler.state_dict() if self.cfg.enable_amp else None,
                        'best_metric_value': self.best_metric_value},
                       filename+".tmp")
            os.replace(filename+".tmp", filename)
            if is_best:
                shutil.copyfile(filename, os.path.join(self.cfg.save_path, 'model', 'model_best.pth'))
                self.logger.info('Best validation {} updated to: {:.4f}'.format(
                    self.cfg.eval_metric, self.best_metric_value))
            self.logger.info('Currently Best {}: {:.4f}'.format(
                self.cfg.eval_metric, self.best_metric_value))
            if self.cfg.save_freq and (self.epoch + 1) % self.cfg.save_freq == 0:
                shutil.copyfile(filename, os.path.join(self.cfg.save_path, 'model', f'epoch_{self.epoch + 1}.pth'))

    def build_model(self):
        model = build_model(self.cfg.model)
        if self.cfg.sync_bn:
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.logger.info(f"Model: \n{model}")
        self.logger.info(f"Num params: {n_parameters}")
        model = create_ddp_model(model.cuda(),
                                 broadcast_buffers=False,
                                 find_unused_parameters=self.cfg.find_unused_parameters)
        return model

    def build_writer(self):
        writer = SummaryWriter(self.cfg.save_path) if comm.is_main_process() else None
        return writer

    def build_train_loader(self):
        train_data = build_dataset(self.cfg.data.train)
        self.categories = train_data.categories
        self.cls2parts = train_data.cls2parts

        if comm.get_world_size() > 1:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)
        else:
            train_sampler = None

        init_fn = partial(
            worker_init_fn, num_workers=self.cfg.num_worker_per_gpu, rank=comm.get_rank(),
            seed=self.cfg.seed) if self.cfg.seed is not None else None

        train_loader = torch.utils.data.DataLoader(train_data,
                                                   batch_size=self.cfg.batch_size_per_gpu,
                                                   shuffle=(train_sampler is None),
                                                   num_workers=self.cfg.num_worker_per_gpu,
                                                   sampler=train_sampler,
                                                   collate_fn=partial(point_collate_fn,
                                                                      max_batch_points=self.cfg.max_batch_points,
                                                                      mix_prob=self.cfg.mix_prob
                                                                      ),
                                                   pin_memory=True,
                                                   worker_init_fn=init_fn,
                                                   drop_last=True,
                                                   persistent_workers=True)
        return train_loader

    def build_val_loader(self):
        val_loader = None
        if self.cfg.evaluate:
            val_data = build_dataset(self.cfg.data.val)
            if comm.get_world_size() > 1:
                val_sampler = torch.utils.data.distributed.DistributedSampler(val_data)
            else:
                val_sampler = None
            val_loader = torch.utils.data.DataLoader(val_data,
                                                     batch_size=self.cfg.batch_size_val_per_gpu,
                                                     shuffle=False,
                                                     num_workers=self.cfg.num_worker_per_gpu,
                                                     pin_memory=True,
                                                     sampler=val_sampler,
                                                     collate_fn=collate_fn)
        return val_loader

    def build_criteria(self):
        return build_criteria(self.cfg.criteria)

    def build_optimizer(self):
        return build_optimizer(self.cfg.optimizer, self.model, self.cfg.param_dicts)

    def build_scheduler(self):
        assert hasattr(self, "optimizer")
        assert hasattr(self, "train_loader")
        self.cfg.scheduler.total_steps = len(self.train_loader) * self.cfg.eval_epoch
        return build_scheduler(self.cfg.scheduler, self.optimizer)

    def build_scaler(self):
        scaler = torch.cuda.amp.GradScaler() if self.cfg.enable_amp else None
        return scaler

    def resume_or_load(self):
        if self.cfg.weight and os.path.isfile(self.cfg.weight):
            self.logger.info(f"Loading weight at: {self.cfg.weight}")
            checkpoint = torch.load(self.cfg.weight, map_location=lambda storage, loc: storage.cuda())
            load_state_info = self.model.load_state_dict(checkpoint['state_dict'], strict=False)
            self.logger.info(f"Missing keys: {load_state_info[0]}")
            if self.cfg.resume:
                self.logger.info(f"Resuming train at eval epoch: {checkpoint['epoch']}")
                self.start_epoch = checkpoint['epoch']
                self.best_metric_value = checkpoint['best_metric_value']
                self.optimizer.load_state_dict(checkpoint['optimizer'])
                self.scheduler.load_state_dict(checkpoint['scheduler'])
                if self.cfg.enable_amp:
                    self.scaler.load_state_dict(checkpoint['scaler'])
        else:
            self.logger.info(f"No weight found at: {self.cfg.weight}")
