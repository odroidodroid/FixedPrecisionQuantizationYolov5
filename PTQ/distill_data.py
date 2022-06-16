

import argparse
import os
import json
import torch
import torch.nn as nn
import copy
import torch.optim as optim
from utils.torch_utils import select_device
from models.yolo import Model
from pathlib import Path
from sampler import *
from mmcv import Config
import yaml
from mmdet.datasets import (build_dataloader, build_dataset, replace_ImageToTensor)

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory


def own_loss(A, B):
    return (A - B).norm()**2 / A.size(0)


class output_hook(object):
    def __init__(self):
        super(output_hook, self).__init__()
        self.outputs = None

    def hook(self, module=None, input=None, output=None):
        tmp_tensor = output
        if self.outputs is None:
            self.outputs = tmp_tensor
        else:
            self.outputs = torch.cat((self.outputs, tmp_tensor), dim=0)

    def clear(self):
        self.outputs = None


def getDistillData(teacher_model, dataloader, work_dir, num_batch, iteration_cnt):
    file_name = f'PTQ_yolov5_distill_'+ str(iteration_cnt)+'.pth'

    print('generating ditilled data')

    hooks, hook_handles, bn_stats, distill_data = [], [], [], []

    for n, m in teacher_model.model.named_modules():
        if n == '24.m.0' or n == '24.m.1' or n == '24.m.2':
            continue        
        if isinstance(m, nn.Conv2d):
            hook = output_hook()
            hooks.append(hook)
            hook_handles.append(m.register_forward_hook(hook.hook))
        if isinstance(m, nn.BatchNorm2d):
            eps = 1e-6
            bn_stats.append(
                (m.running_mean.detach().clone().flatten().cuda(),
                 torch.sqrt(m.running_var +
                            eps).detach().clone().flatten().cuda()))

    assert len(hooks) == len(bn_stats)
    teacher_model = nn.DataParallel(teacher_model, device_ids=[0])
    teacher_model.eval()

    logs = []
    for i, gaussian_data in enumerate(dataloader):
        if i == num_batch:
            break
        logs.append({})
        # Uniform initilizaition and normalize to 0
        size = (1, 3, 640, 640)
        gaussian_data['img'] = [
            (torch.randint(high=255, size=size) - 128).float().cuda() / 5418.75
        ]

        im = torch.tensor(gaussian_data['img'][0].clone().detach().requires_grad_(True))
        optimizer = optim.Adam([gaussian_data['img'][0]], lr=0.1)
        # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, min_lr=1e-4, verbose=False, patience=100)
        input_mean = torch.zeros(1, 3).cuda()
        input_std = torch.ones(1, 3).cuda()

        for it in range(iteration_cnt):
            teacher_model.zero_grad()
            optimizer.zero_grad()
            for hook in hooks:
               hook.clear()
            output = teacher_model(im)
            mean_loss = 0
            std_loss = 0
            for cnt, (bn_stat, hook) in enumerate(zip(bn_stats, hooks)):
                tmp_output = hook.outputs
                bn_mean, bn_std = bn_stat[0], bn_stat[1]
                tmp_mean = torch.mean(tmp_output.view(tmp_output.size(0),
                                                      tmp_output.size(1), -1),
                                      dim=2)
                tmp_std = torch.sqrt(
                    torch.var(tmp_output.view(tmp_output.size(0),
                                              tmp_output.size(1), -1),
                              dim=2) + eps)
                mean_loss += own_loss(bn_mean, tmp_mean)
                std_loss += own_loss(bn_std, tmp_std)
                # print(cnt, mean_loss.item(), std_loss.item())
            tmp_mean = torch.mean(im.view(size[0], 3, -1),
                                  dim=2)
            tmp_std = torch.sqrt(
                torch.var(im.view(size[0], 3, -1), dim=2)
                + eps)
            mean_loss += own_loss(input_mean, tmp_mean)
            std_loss += own_loss(input_std, tmp_std)
            total_loss = mean_loss + std_loss
            total_loss.backward()
            # print(it, mean_loss.item(), std_loss.item())
            logs[-1][it] = copy.deepcopy(total_loss.item())
            optimizer.step()
            # scheduler.step(total_loss.item())
            # if total_loss <= (len(hooks) + 1) * 10:
            # 	break
        im = im.detach().clone()
        distill_data.append(gaussian_data)

    for handle in hook_handles:
        handle.remove()

    torch.save(distill_data, file_name)
    json.dump(logs, open(f'loss_log.json', 'w'))

    return distill_data


def parse_args():
    parser = argparse.ArgumentParser(description='test detector')
    parser.add_argument('--dataset_path', type=str, default='/home/youngjin/datasets/coco/val',help='test dataset path')
    parser.add_argument('--data_config', type=str, default=ROOT / '../dataset/coco.yaml')
    parser.add_argument('--model_config', type=str, default=ROOT/ 'models/yolov5l.yaml')
    parser.add_argument('--hyp', type=str, default=ROOT/ '../dataset/hyps/hyp.scratch-low.yaml')
    parser.add_argument('--data', type=str, default=ROOT / '../config/distill_data_config.py')
    parser.add_argument('--device', type=str, default='0,1')
    parser.add_argument('--distribuited', default=True)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--checkpoint', type=str, default=ROOT / '../checkpoints/yolov5l.pt',help='checkpoint file')
    parser.add_argument('--tmpdir', type=str, default=ROOT / '../../runs/distill_data', help='tmp dir for writing some results')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--cudnn_benchmark',default=True)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def main():
    args = parse_args()
    # set cudnn_benchmark
    # build the dataloader
    # TODO: support multiple images per gpu (only minor changes are needed)

    # Hyperparameters
    with open(args.hyp, errors='ignore') as f:
        hyp = yaml.safe_load(f)  # load hyps dict

    cfg = Config.fromfile(args.data)

    test_dataloader_default_args = dict(
        samples_per_gpu=1, workers_per_gpu=2, dist=args.distribuited, shuffle=False)

    test_loader_cfg = {
        **test_dataloader_default_args,
        **cfg.data.get('test_dataloader', {})
    }

    dataset = build_dataset(cfg.data.val)
    data_loader = build_dataloader(dataset, **test_loader_cfg)

    # build the model and load checkpoint
    device = select_device(args.device)
    #model = DetectMultiBackend(weights=args.checkpoint, device=device, dnn=False, data=args.data_config, fp16=False)
    model = Model(args.model_config, ch=3, nc=80, anchors=hyp.get('anchors')).to(device)

    getDistillData(teacher_model=model,
                   dataloader=data_loader,
                   work_dir='',
                   num_batch=1,
                   iteration_cnt=500)


if __name__ == '__main__':
    main()
