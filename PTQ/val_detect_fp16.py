# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Validate a trained YOLOv5 model accuracy on a custom dataset

Usage:
    $ python path/to/val.py --weights yolov5s.pt --data coco128.yaml --img 640

Usage - formats:
    $ python path/to/val.py --weights yolov5s.pt                 # PyTorch
                                      yolov5s.torchscript        # TorchScript
                                      yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                      yolov5s.xml                # OpenVINO
                                      yolov5s.engine             # TensorRT
                                      yolov5s.mlmodel            # CoreML (macOS-only)
                                      yolov5s_saved_model        # TensorFlow SavedModel
                                      yolov5s.pb                 # TensorFlow GraphDef
                                      yolov5s.tflite             # TensorFlow Lite
                                      yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
"""

import argparse
import json
import os
import sys
from pathlib import Path
from threading import Thread
from hypothesis import target

import numpy as np
import torch
from tqdm import tqdm

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.callbacks import Callbacks
from utils.datasets import *
from utils.general import (LOGGER, check_dataset, check_img_size, check_requirements, check_yaml,
                           coco80_to_coco91_class, colorstr, increment_path, non_max_suppression, print_args,
                           scale_coords, xywh2xyxy, xyxy2xywh, xywh2xyxy_custom, xywh2xyxy_custom2,coco91_to_coco80_class)
from utils.metrics import ConfusionMatrix, ap_per_class, box_iou
from utils.plots import output_to_target, plot_images, plot_val_study
from utils.plots import *
from utils.torch_utils import select_device, time_sync
import cv2


def save_one_txt(predn, save_conf, shape, file):
    # Save one txt result
    gn = torch.tensor(shape)[[1, 0, 1, 0]]  # normalization gain whwh
    for *xyxy, conf, cls in predn.tolist():
        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
        with open(file, 'a') as f:
            f.write(('%g ' * len(line)).rstrip() % line + '\n')


def save_one_image(predn, names, save_conf, shape, file, img_id, im0) :

    #im0 = cv2.imread('/home/youngjin/datasets/coco/val/images/' + img_id + '.jpg')
    annotator = Annotator(im0, line_width=3, example=str(names))
    for *xyxy, conf, cls in predn.tolist() :
        c = int(cls)
        label = names[c]
        annotator.box_label(xyxy, label, color=(c, True))
    im0 = annotator.result()
    cv2.imwrite(file, im0)


def save_targets(targets, names, save_conf, shape, file, img_id, im0) :

    #im0 = cv2.imread('/home/youngjin/datasets/coco/val/images/' + img_id + '.jpg')
    annotator = Annotator(im0, line_width=3, example=str(names))
    targets = targets.view(-1,5)
    for cls, *xyxy in targets.tolist() :
        c = int(cls)
        label = names[c]
        annotator.box_label(xyxy, label, color=(c, True))
    im0 = annotator.result()
    cv2.imwrite(file, im0)



def save_one_json(predn, jdict, path, class_map):
    # Save one JSON result {"image_id": 42, "category_id": 18, "bbox": [258.15, 41.29, 348.26, 243.78], "score": 0.236}
    image_id = int(path.stem) if path.stem.isnumeric() else path.stem
    box = xyxy2xywh(predn[:, :4])  # xywh
    box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
    for p, b in zip(predn.tolist(), box.tolist()):
        jdict.append({
            'image_id': image_id,
            'category_id': class_map[int(p[5])],
            'bbox': [round(x, 3) for x in b],
            'score': round(p[4], 5)})


def process_batch(detections, labels, iouv):
    """
    Return correct predictions matrix. Both sets of boxes are in (x1, y1, x2, y2) format.
    Arguments:
        detections (Array[N, 6]), x1, y1, x2, y2, conf, class
        labels (Array[M, 5]), class, x1, y1, x2, y2
    Returns:
        correct (Array[N, 10]), for 10 IoU levels
    """
    correct = torch.zeros(detections.shape[0], iouv.shape[0], dtype=torch.bool, device=iouv.device)
    iou = box_iou(labels[:, 1:], detections[:, :4])
    x = torch.where((iou >= iouv[0]) & (labels[:, 0:1] == detections[:, 5]))  # IoU above threshold and classes match
    if x[0].shape[0]:
        matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()  # [label, detection, iou]
        if x[0].shape[0] > 1:
            matches = matches[matches[:, 2].argsort()[::-1]]
            matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
            # matches = matches[matches[:, 2].argsort()[::-1]]
            matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
        matches = torch.from_numpy(matches).to(iouv.device)
        correct[matches[:, 1].long()] = matches[:, 2:3] >= iouv
    return correct


@torch.no_grad()
def run(
        data,
        weights= ROOT / '../checkpoints/yolov5l.pt',  # model.pt path(s)
        source='/home/youngjin/datasets/coco/val',
        hyp='',
        evaluate=True,
        batch_size=1,  # batch size
        imgsz=640,  # inference size (pixels)
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        task='val',
        agnostic_nms=False,  # class-agnostic NMS
        max_det=1000,  # maximum detections per image
        conf_thres=0.45,  # confidence threshold
        iou_thres=0.25,  # NMS IoU threshold
        device='0,1',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        workers=8,  # max dataloader workers (per RANK in DDP mode)
        single_cls=False,  # treat as single-class dataset
        augment=False,  # augmented inference
        verbose=False,  # verbose output
        save_txt=True,  # save results to *.txt
        save_img=True,
        save_hybrid=False,  # save label+prediction hybrid results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_json=True,  # save a COCO-JSON results file
        project=ROOT / '../../runs/quant_val_detect',  # save to project/name
        name='exp',  # save to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        model=None,
        save_dir=Path(''),
        plots=False,
        callbacks=Callbacks(),
        compute_loss=None,
):
    # Initialize/load model and set device
    training = model is not None
    if training:  # called by train.py
        device, pt, jit, engine = next(model.parameters()).device, True, False, False  # get model device, PyTorch model
        half &= device.type != 'cpu'  # half precision only supported on CUDA
        model.half() if half else model.float()
    else:  # called directly
        device = select_device(device, batch_size=batch_size)

        # Directories
        save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
        (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir
        (save_dir / 'images' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir
        (save_dir / 'targets' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

        # Load model
        model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)

        stride, pt, jit, engine = model.stride, model.pt, False, False
        imgsz = check_img_size(imgsz, s=stride)  # check image size
        half = model.fp16  # FP16 supported on limited backends with CUDA
        gs = max(int(model.stride), 32)  # grid size (max stride)


        data_loader, dataset = create_dataloader_custom(image_path=source+'/images',
                                                        label_path=source+'/labels',
                                                        imgsz=imgsz,
                                                        batch_size=1,
                                                        stride=stride,
                                                        workers=workers)



        # Load model



        if engine:
            batch_size = model.batch_size
        else:
            device = model.device
            if not pt:
                batch_size = 1  # export.py models default to batch-size 1
                LOGGER.info(f'Forcing --batch-size 1 square inference (1,3,{imgsz},{imgsz}) for non-PyTorch models')

        # Data
        data = check_dataset(data)  # check

    # Configure
    model.eval()
    cuda = device.type != 'cpu'
    nc = 1 if single_cls else int(data['nc'])  # number of classes
    iouv = torch.linspace(0.5, 0.95, 10, device=device)  # iou vector for mAP@0.5:0.95
    niou = iouv.numel()

    seen = 0
    confusion_matrix = ConfusionMatrix(nc=nc)
    names = {k: v for k, v in enumerate(model.names if hasattr(model, 'names') else model.module.names)}
    #class_map = coco80_to_coco91_class() if is_coco else list(range(1000))
    s = ('%20s' + '%11s' * 6) % ('Class', 'Images', 'Labels', 'P', 'R', 'mAP@.5', 'mAP@.5:.95')
    dt, p, r, f1, mp, mr, map50, map = [0.0, 0.0, 0.0], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    loss = torch.zeros(3, device=device)
    jdict, stats, ap, ap_class = [], [], [], []
    callbacks.run('on_val_start')
    pbar = tqdm(dataset)
    eps = 1e-16

    for batch_i, (img, im0, targets, paths, shapes, img_id) in enumerate(pbar) :
        callbacks.run('on_val_batch_start')
        t1 = time_sync()
        img = torch.from_numpy(img).to(device)
        if not (targets is None) :
            targets = torch.from_numpy(targets).to(device)
        paths = source + '/images/' + img_id + '.jpg'
        img = img.view(1, img.shape[0], img.shape[1],img.shape[2])
            #targets = torch.from_numpy(targets).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255  # 0 - 255 to 0.0 - 1.0
        nb, _, height, width = img.shape  # batch size, channels, height, width
        t2 = time_sync()
        dt[0] += t2 - t1

        # Inference
        out = model(img, augment=augment)  # inference, loss outputs
        dt[1] += time_sync() - t2

        # Loss
        #if compute_loss:
        #    loss += compute_loss([x.float() for x in train_out], targets)[1]  # box, obj, cls

        # NMS
        t3 = time_sync()
        out = non_max_suppression(out, conf_thres, iou_thres, classes, agnostic_nms, max_det)
        dt[2] += time_sync() - t3

        seen += 1
        if evaluate and not (targets is None): 
            for si, pred in enumerate(out):
                pred = pred.view(-1, 6)
                cat_ids, bboxes = coco91_to_coco80_class(targets) 
                nl, npr = cat_ids.shape[0], pred.shape[0]  # number of labels, predictions
                correct = torch.zeros(npr, niou, dtype=torch.bool, device=device)  # init

                if npr == 0:
                    if nl:
                        stats.append((correct, *torch.zeros((3, 0), device=device)))
                    continue

                # Predictions
                predn = pred.clone()
                predn[:, :4] = scale_coords(img.shape[2:], predn[:, :4], im0.shape)  # native-space pred

                # Evaluate
                if nl:
                    bboxes = xywh2xyxy_custom2(bboxes)
                    labelsn = torch.cat((cat_ids, bboxes), 1)  # native-space labels
                    correct = process_batch(predn, labelsn, iouv)
                    if plots:
                        confusion_matrix.process_batch(predn, labelsn)
                stats.append((correct, pred[:, 4], pred[:, 5], cat_ids[:, 0]))  # (correct, conf, pcls, tcls)
            
            # Save targets
            save_targets(labelsn, names, save_conf, shapes,
            file=save_dir / 'targets' / (img_id + '.jpg'), img_id=img_id,
            im0=copy(im0))

            # Save/log
            if save_txt:
                save_one_txt(predn, save_conf, shapes, file=save_dir / 'labels' / (img_id + '.txt'))
            if save_img :
                save_one_image(predn, names, save_conf, shapes, 
                file=save_dir / 'images' / (img_id + '.jpg'), img_id=img_id,
                im0=im0)
        
            if save_json:
                pass
                #save_one_json(predn, jdict, path, class_map)  # append to COCO-JSON dictionary
            callbacks.run('on_val_image_end', pred, predn, paths, names, img)


        # Plot images
        if plots and batch_i < 3:
            f = save_dir / f'val_batch{batch_i}_labels.jpg'  # labels
            Thread(target=plot_images, args=(im0, targets, paths, f, names), daemon=True).start()
            f = save_dir / f'val_batch{batch_i}_pred.jpg'  # predictions
            Thread(target=plot_images, args=(im0, output_to_target(out), paths, f, names), daemon=True).start()

        callbacks.run('on_val_batch_end')

    # Compute metric
    stats = [torch.cat(x, 0).cpu().numpy() for x in zip(*stats)]  # to numpy
    tp, conf, pred_cls, target_cls = stats

    # i = np.argsort(-conf.cpu().detach().numpy())
    # tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]
    n_l = target_cls.shape[0]
    fpc = (1-tp).cumsum()[-1]
    tpc = tp.cumsum()[-1]

    recall = tpc / (n_l + eps)
    precision = tpc / (tpc + fpc)

    print('cumsum recall : {}, precision : {}'.format(recall, precision))


    # # Compute metrics
    # stats = [torch.cat(x, 0).cpu().numpy() for x in zip(*stats)]  # to numpy
    # if len(stats) and stats[0].any():
    #     tp, fp, p, r, f1, ap, ap_class = ap_per_class(*stats, plot=plots, save_dir=save_dir, names=names)
    #     ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
    #     mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
    #     nt = np.bincount(stats[3].astype(np.int64), minlength=nc)  # number of targets per class
    # else:
    #     nt = torch.zeros(1)

    # # Print results
    # pf = '%20s' + '%11i' * 2 + '%11.3g' * 4  # print format
    # LOGGER.info(pf % ('all', seen, nt.sum(), mp, mr, map50, map))

    # # Print results per class
    # if (verbose or (nc < 50 and not training)) and nc > 1 and len(stats):
    #     for i, c in enumerate(ap_class):
    #         LOGGER.info(pf % (names[c], seen, nt[c], p[i], r[i], ap50[i], ap[i]))

    # Print speeds
    t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    if not training:
        shape = (batch_size, 3, imgsz, imgsz)
        LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {shape}' % t)

    # # Plots
    # if plots:
    #     confusion_matrix.plot(save_dir=save_dir, names=list(names.values()))
    #     callbacks.run('on_val_end')

    # # Save JSON
    # if save_json and len(jdict):
    #     w = Path(weights[0] if isinstance(weights, list) else weights).stem if weights is not None else ''  # weights
    #     anno_json = str(Path(data.get('path', '../coco')) / 'annotations/instances_val2017.json')  # annotations json
    #     pred_json = str(save_dir / f"{w}_predictions.json")  # predictions json
    #     LOGGER.info(f'\nEvaluating pycocotools mAP... saving {pred_json}...')
    #     with open(pred_json, 'w') as f:
    #         json.dump(jdict, f)

    #     try:  # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
    #         check_requirements(['pycocotools'])
    #         from pycocotools.coco import COCO
    #         from pycocotools.cocoeval import COCOeval

    #         anno = COCO(anno_json)  # init annotations api
    #         pred = anno.loadRes(pred_json)  # init predictions api
    #         eval = COCOeval(anno, pred, 'bbox')
    #         #if is_coco:
    #         #    eval.params.imgIds = [int(Path(x).stem) for x in dataloader.dataset.im_files]  # image IDs to evaluate
    #         eval.evaluate()
    #         eval.accumulate()
    #         eval.summarize()
    #         map, map50 = eval.stats[:2]  # update results (mAP@0.5:0.95, mAP@0.5)
    #     except Exception as e:
    #         LOGGER.info(f'pycocotools unable to run: {e}')

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default=ROOT / '../dataset/coco.yaml', help='dataset.yaml path')
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5l.pt', help='model.pt path(s)')
    parser.add_argument('--source', default='/home/youngjin/datasets/coco/val')
    parser.add_argument('--hyp', default='../dataset/hyps/hyp.scratch-low.yaml')
    parser.add_argument('--batch-size', type=int, default=2, help='batch size')
    parser.add_argument('--evaluate', default=True)
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+',type=int, default=[640], help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--task', default='val', help='train, val, test, speed or study')
    parser.add_argument('--device', default='0,1', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--workers', type=int, default=8, help='max dataloader workers (per RANK in DDP mode)')
    parser.add_argument('--single-cls', action='store_true', help='treat as single-class dataset')
    parser.add_argument('--augment', default=False, help='augmented inference')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--verbose', action='store_true', help='report mAP by class')
    parser.add_argument('--save-txt', default=True, help='save results to *.txt')
    parser.add_argument('--save-img', default=True)
    parser.add_argument('--save-hybrid', default=False, help='save label+prediction hybrid results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-json', default=False, help='save a COCO-JSON results file')
    parser.add_argument('--project', default=ROOT / '../../runs/val_detect', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--half', default=True, help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    opt.data = check_yaml(opt.data)  # check YAML
    #opt.save_json |= opt.data.endswith('coco.yaml')
    #opt.save_txt |= opt.save_hybrid
    print_args(vars(opt))
    return opt


def main(opt):
    check_requirements(requirements=ROOT / 'requirements.txt', exclude=('tensorboard', 'thop'))

    if opt.task in ('train', 'val', 'test'):  # run normally
        if opt.conf_thres > 0.001:  # https://github.com/ultralytics/yolov5/issues/1466
            LOGGER.info(f'WARNING: confidence threshold {opt.conf_thres} >> 0.001 will produce invalid mAP values.')
        run(**vars(opt))

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
