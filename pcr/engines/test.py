import os
import time
import numpy as np
import pickle
import torch
import torch.nn.functional as F
from ..utils.registry import Registry
from ..utils.logger import get_root_logger
from ..utils.misc import AverageMeter, intersection_and_union, intersection_and_union_gpu, make_dirs
from ..datasets.utils import collate_fn

TEST = Registry("test")


@TEST.register_module()
class SegmentationTest(object):
    """SegmentationTest
    for large outdoor point cloud with need voxelize (s3dis)
    """

    def __call__(self, cfg, test_loader, model):
        test_dataset = test_loader.dataset
        logger = get_root_logger()
        logger.info('>>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>')

        batch_time = AverageMeter()
        intersection_meter = AverageMeter()
        union_meter = AverageMeter()
        target_meter = AverageMeter()
        model.eval()

        save_path = os.path.join(cfg.save_path, "result", "test_epoch{}".format(cfg.epochs))
        make_dirs(save_path)
        if "ScanNet" in cfg.dataset_type:
            sub_path = os.path.join(save_path, "submit")
            make_dirs(sub_path)
        pred_save, label_save = [], []
        for idx in range(len(test_dataset)):
            end = time.time()
            data_name = test_dataset.get_data_name(idx)
            pred_save_path = os.path.join(save_path, '{}_pred.npy'.format(data_name))
            label_save_path = os.path.join(save_path, '{}_label.npy'.format(data_name))
            if os.path.isfile(pred_save_path) and os.path.isfile(label_save_path):
                logger.info('{}/{}: {}, loaded pred and label.'.format(idx + 1, len(test_dataset), data_name))
                pred, label = np.load(pred_save_path), np.load(label_save_path)
            else:
                data_dict_list, label = test_dataset[idx]
                pred = torch.zeros((label.size, cfg.data.num_classes)).cuda()
                batch_num = int(np.ceil(len(data_dict_list) / cfg.batch_size_test))
                for i in range(batch_num):
                    s_i, e_i = i * cfg.batch_size_test, min((i + 1) * cfg.batch_size_test, len(data_dict_list))
                    input_dict = collate_fn(data_dict_list[s_i:e_i])
                    for key in input_dict.keys():
                        input_dict[key] = input_dict[key].cuda(non_blocking=True)
                    idx_part = input_dict["index"]
                    with torch.no_grad():
                        pred_part = model(input_dict)  # (n, k)
                        pred_part = F.softmax(pred_part, -1)
                    if cfg.empty_cache:
                        torch.cuda.empty_cache()
                    bs = 0
                    for be in input_dict["offset"]:
                        pred[idx_part[bs: be], :] += pred_part[bs: be]
                        bs = be
                    logger.info('Test: {} {}/{}, Batch: {batch_idx}/{batch_num}'.format(
                        data_name, idx + 1, len(test_dataset), batch_idx=i, batch_num=batch_num))
                if cfg.save_probs:
                    probs = pred.data.cpu().numpy().copy()
                pred = pred.max(1)[1].data.cpu().numpy()
            intersection, union, target = intersection_and_union(pred, label, cfg.data.num_classes,
                                                                 cfg.data.ignore_label)
            intersection_meter.update(intersection)
            union_meter.update(union)
            target_meter.update(target)

            mask = union != 0
            iou_class = intersection / (union + 1e-10)
            iou = np.mean(iou_class[mask])
            acc = sum(intersection) / (sum(target) + 1e-10)

            m_iou = np.mean(intersection_meter.sum / (union_meter.sum + 1e-10))
            m_acc = np.mean(intersection_meter.sum / (target_meter.sum + 1e-10))

            batch_time.update(time.time() - end)
            logger.info('Test: {} [{}/{}]-{} '
                        'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                        'Accuracy {acc:.4f} ({m_acc:.4f}) '
                        'mIoU {iou:.4f} ({m_iou:.4f})'.format(data_name, idx + 1, len(test_dataset), label.size,
                                                              batch_time=batch_time, acc=acc, m_acc=m_acc,
                                                              iou=iou, m_iou=m_iou))
            probs = probs if cfg.save_probs else pred
            pred_save.append(probs)
            label_save.append(label)
            np.save(pred_save_path, pred)
            np.save(label_save_path, label)
            if "ScanNet" in cfg.dataset_type:
                np.savetxt(os.path.join(save_path, "submit", '{}.txt'.format(data_name)),
                           test_dataset.class2id[pred].reshape([-1, 1]), fmt="%d")

        with open(os.path.join(save_path, "pred.pickle"), 'wb') as handle:
            pickle.dump({'pred': pred_save}, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(os.path.join(save_path, "label.pickle"), 'wb') as handle:
            pickle.dump({'label': label_save}, handle, protocol=pickle.HIGHEST_PROTOCOL)

        iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
        accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
        mIoU = np.mean(iou_class)
        mAcc = np.mean(accuracy_class)
        allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)

        logger.info('Val result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}'.format(mIoU, mAcc, allAcc))
        for i in range(cfg.data.num_classes):
            logger.info('Class_{idx} - {name} Result: iou/accuracy {iou:.4f}/{accuracy:.4f}'.format(
                idx=i, name=cfg.data.names[i], iou=iou_class[i], accuracy=accuracy_class[i]))
        logger.info('<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<')


@TEST.register_module()
class ClassificationTest(object):
    """ClassificationTest
    for classification dataset (modelnet40), containing multi scales voting
    """

    def __init__(self,
                 scales=(0.9, 0.95, 1, 1.05, 1.1),
                 shuffle=False):
        self.scales = scales
        self.shuffle = shuffle

    def __call__(self, cfg, test_loader, model):
        logger = get_root_logger()
        logger.info('>>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>')
        batch_time = AverageMeter()
        intersection_meter = AverageMeter()
        union_meter = AverageMeter()
        target_meter = AverageMeter()
        model.eval()

        for i, input_dict in enumerate(test_loader):
            for key in input_dict.keys():
                input_dict[key] = input_dict[key].cuda(non_blocking=True)
            coord = input_dict["coord"]
            feat = input_dict["feat"]
            target = input_dict["label"]
            offset = input_dict["offset"]
            end = time.time()
            output = torch.zeros([offset.shape[0], cfg.data.num_classes], dtype=torch.float32).cuda()
            for scale in self.scales:
                coord_temp, feat_temp = [], []
                for k in range(offset.shape[0]):
                    if k == 0:
                        s_k, e_k, cnt = 0, offset[0], offset[0]
                    else:
                        s_k, e_k, cnt = offset[k - 1], offset[k], offset[k] - offset[k - 1]
                    coord_part, feat_part = coord[s_k:e_k, :], feat[s_k:e_k, :]
                    coord_part *= scale
                    idx = np.arange(coord_part.shape[0])
                    if self.shuffle:
                        np.random.shuffle(idx)
                    coord_temp.append(coord_part[idx]), feat_temp.append(feat_part[idx])
                coord_temp, feat_temp = torch.cat(coord_temp, 0), torch.cat(feat_temp, 0)
                with torch.no_grad():
                    output_part = model(dict(coord=coord_temp, feat=feat_temp, offset=offset))
                output += output_part
            output = output.max(1)[1]
            intersection, union, target = intersection_and_union_gpu(output, target, cfg.data.num_classes,
                                                                     cfg.data.ignore_label)
            intersection, union, target = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy()
            intersection_meter.update(intersection), union_meter.update(union), target_meter.update(target)

            accuracy = sum(intersection_meter.val) / (sum(target_meter.val) + 1e-10)
            batch_time.update(time.time() - end)

            logger.info('Test: [{}/{}] '
                        'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                        'Accuracy {accuracy:.4f} '.format(i + 1, len(test_loader),
                                                          batch_time=batch_time,
                                                          accuracy=accuracy))

        iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
        accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
        mIoU = np.mean(iou_class)
        mAcc = np.mean(accuracy_class)
        allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)
        logger.info('Val result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(mIoU, mAcc, allAcc))

        for i in range(cfg.data.num_classes):
            logger.info('Class_{idx} - {name} Result: iou/accuracy {iou:.4f}/{accuracy:.4f}'.format(
                idx=i, name=cfg.data.names[i], iou=iou_class[i], accuracy=accuracy_class[i]))
        logger.info('<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<')


@TEST.register_module()
class PartSegmentationTest(object):
    """PartSegmentationTest
    """

    def __call__(self, cfg, test_loader, model):
        test_dataset = test_loader.dataset
        logger = get_root_logger()
        logger.info('>>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>')

        batch_time = AverageMeter()
        intersection_meter = AverageMeter()
        union_meter = AverageMeter()
        target_meter = AverageMeter()

        num_categories = len(test_loader.dataset.categories)
        iou_category, iou_count = np.zeros(num_categories), np.zeros(num_categories)
        model.eval()

        save_path = os.path.join(cfg.save_path, "result", "test_epoch{}".format(cfg.epochs))
        make_dirs(save_path)

        for idx in range(len(test_dataset)):
            end = time.time()
            data_name = test_dataset.get_data_name(idx)

            data_dict_list, label = test_dataset[idx]
            pred = torch.zeros((label.size, cfg.data.num_classes)).cuda()
            batch_num = int(np.ceil(len(data_dict_list) / cfg.batch_size_test))
            for i in range(batch_num):
                s_i, e_i = i * cfg.batch_size_test, min((i + 1) * cfg.batch_size_test, len(data_dict_list))
                input_dict = collate_fn(data_dict_list[s_i:e_i])
                for key in input_dict.keys():
                    input_dict[key] = input_dict[key].cuda(non_blocking=True)
                with torch.no_grad():
                    pred_part = model(input_dict)
                    pred_part = F.softmax(pred_part, -1)
                if cfg.empty_cache:
                    torch.cuda.empty_cache()
                pred_part = pred_part.reshape(-1, label.size, cfg.data.num_classes)
                pred = pred + pred_part.total(dim=0)
                logger.info('Test: {} {}/{}, Batch: {batch_idx}/{batch_num}'.format(
                    data_name, idx + 1, len(test_dataset), batch_idx=i, batch_num=batch_num))
            pred = pred.max(1)[1].data.cpu().numpy()

            category_index = data_dict_list[0]["cls_token"]
            category = test_loader.dataset.categories[category_index]
            parts_idx = test_loader.dataset.category2part[category]
            parts_iou = np.zeros(len(parts_idx))
            for j, part in enumerate(parts_idx):
                if (np.sum(label == part) == 0) and (np.sum(pred == part) == 0):
                    parts_iou[j] = 1.0
                else:
                    i = (label == part) & (pred == part)
                    u = (label == part) | (pred == part)
                    parts_iou[j] = np.sum(i) / (np.sum(u) + 1e-10)
            iou_category[category_index] += parts_iou.mean()
            iou_count[category_index] += 1

            batch_time.update(time.time() - end)
            logger.info('Test: {} [{}/{}] '
                        'Batch {batch_time.val:.3f} '
                        '({batch_time.avg:.3f}) '.format(
                data_name, idx + 1, len(test_loader), batch_time=batch_time))

        ins_mIoU = iou_category.sum() / (iou_count.sum() + 1e-10)
        cat_mIoU = (iou_category / (iou_count + 1e-10)).mean()
        logger.info('Val result: ins.mIoU/cat.mIoU {:.4f}/{:.4f}.'.format(ins_mIoU, cat_mIoU))
        for i in range(num_categories):
            logger.info('Class_{idx}-{name} Result: iou_cat/num_sample {iou_cat:.4f}/{iou_count:.4f}'.format(
                idx=i, name=test_loader.dataset.categories[i],
                iou_cat=iou_category[i] / (iou_count[i] + 1e-10),
                iou_count=int(iou_count[i])))
        logger.info('<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<')


@TEST.register_module()
class PartSegmentationTestV2(object):
    """PartSegmentationTest
    """

    def __call__(self, cfg, test_loader, model):
        from ..engines.partseg import part_seg_refinement, get_ins_mious
        test_dataset = test_loader.dataset
        logger = get_root_logger()
        logger.info('>>>>>>>>>>>>>>>> Start Evaluation V2 >>>>>>>>>>>>>>>>')

        batch_time = AverageMeter()
        intersection_meter = AverageMeter()
        union_meter = AverageMeter()
        target_meter = AverageMeter()

        # num_categories = len(test_loader.dataset.categories)
        # iou_category, iou_count = np.zeros(num_categories), np.zeros(num_categories)
        model.eval()

        save_path = os.path.join(cfg.save_path, "result", "test_epoch{}".format(cfg.epochs))
        make_dirs(save_path)

        batch_size = 1
        self.cls2parts = test_dataset.cls2parts

        cls_mious = torch.zeros(len(self.cls2parts), dtype=torch.float32).cuda(non_blocking=True)
        cls_nums = torch.zeros(len(self.cls2parts), dtype=torch.int32).cuda(non_blocking=True)
        ins_miou_list = []

        for idx in range(len(test_dataset)):
            end = time.time()
            data_name = test_dataset.get_data_name(idx)

            data_dict_list, label = test_dataset[idx]
            data_size = label.size
            label = torch.from_numpy(label).cuda(non_blocking=True)
            pred = torch.zeros((data_size, cfg.data.num_classes)).cuda()
            batch_num = int(np.ceil(len(data_dict_list) / cfg.batch_size_test))
            for i in range(batch_num):
                s_i, e_i = i * cfg.batch_size_test, min((i + 1) * cfg.batch_size_test, len(data_dict_list))
                input_dict = collate_fn(data_dict_list[s_i:e_i])
                for key in input_dict.keys():
                    input_dict[key] = input_dict[key].cuda(non_blocking=True)
                with torch.no_grad():
                    pred_part = model(input_dict)
                    pred_part = F.softmax(pred_part, -1)
                if cfg.empty_cache:
                    torch.cuda.empty_cache()
                pred_part = pred_part.reshape(-1, data_size, cfg.data.num_classes)
                pred = pred + pred_part.sum(dim=0)
            pred = pred.max(1)[1]

            pred = pred.reshape(batch_size, -1)
            label = label.reshape(batch_size, -1)
            coord = data_dict_list[0]["coord"].reshape(batch_size, -1, data_dict_list[0]["coord"].shape[-1]).cuda(non_blocking=True)

            part_seg_refinement(pred, coord, data_dict_list[0]["cls_token"], self.cls2parts)
            batch_ins_mious = get_ins_mious(pred, label, data_dict_list[0]["cls_token"], self.cls2parts)
            ins_miou_list += batch_ins_mious
            # per category iou at each batch_size:
            for shape_idx in range(batch_size):  # sample_idx
                cur_gt_label = int(data_dict_list[0]["cls_token"][shape_idx].cpu().numpy())
                # add the iou belongs to this cat
                cls_mious[cur_gt_label] += batch_ins_mious[shape_idx]
                cls_nums[cur_gt_label] += 1

            batch_time.update(time.time() - end)
            logger.info('Test: {} [{}/{}] '
                        'Batch {batch_time.val:.3f} '
                        '({batch_time.avg:.3f}) '.format(
                data_name, idx + 1, len(test_dataset), batch_time=batch_time))

        ins_mious_sum, count = torch.sum(torch.stack(ins_miou_list)), torch.tensor(len(ins_miou_list)).cuda()

        for cat_idx in range(len(self.cls2parts)):
            # indicating this cat is included during previous iou appending
            if cls_nums[cat_idx] > 0:
                cls_mious[cat_idx] = cls_mious[cat_idx] / cls_nums[cat_idx]

        ins_miou = ins_mious_sum/count
        cls_miou = torch.mean(cls_mious)

        logger.info('Test result: Ins_mIoU/Cls_mIoU {:.4f}/{:.4f}.'.format(
            ins_miou, cls_miou))
        for i in range(len(self.cls2parts)):
            logger.info('Class_{idx}-{name} Result: iou {iou:.4f}'.format(
                idx=i, name=test_loader.dataset.categories[i], iou=cls_mious[i]))

        logger.info('<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<')
