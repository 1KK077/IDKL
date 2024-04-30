import torch
import math
import torch.nn as nn
import numpy as np
import os
from apex import amp
from ignite.engine import Engine
from ignite.engine import Events
from torch.autograd import no_grad
from torch.nn import functional as F
import torchvision.transforms as T
import cv2
from torchvision.io.image import read_image
from PIL import Image
from torchvision.transforms.functional import normalize, resize, to_pil_image

from torchvision import transforms
from grad_cam.utils import GradCAM, show_cam_on_image, center_crop_img
import copy
from torch.optim.lr_scheduler import LambdaLR


# import torch
# import numpy as np
# import os
# from apex import amp
# from ignite.engine import Engine
# from ignite.engine import Events
# from torch.autograd import no_grad
# from torchvision import transforms
# from PIL import Image
# import cv2
# from grad_cam.utils import GradCAM, show_cam_on_image, center_crop_img
# from thop import profile
# from thop import clever_format
#
# from utils.calc_acc import calc_acc
# from torch.nn import functional as F
# #import objgraph


def some_function(epoch, initial_weight_decay):
    if epoch > 15:
        new_weight_decay = initial_weight_decay/100
    elif epoch > 5 and epoch <= 15:
        new_weight_decay = initial_weight_decay*1/10
    else:
        new_weight_decay = initial_weight_decay
    return new_weight_decay

def create_train_engine(model, optimizer, non_blocking=False):
    device = torch.device("cuda") #"cuda", torch.cuda.current_device()

    def _process_func(engine, batch):
        model.train()
        #model.eval()

        data, labels, cam_ids, img_paths, img_ids = batch
        epoch = engine.state.epoch
        iteration = engine.state.iteration

        data = data.to(device, non_blocking=non_blocking)
        labels = labels.to(device, non_blocking=non_blocking)
        cam_ids = cam_ids.to(device, non_blocking=non_blocking)

        warmup = False
        if warmup == True: #学习率warmup
            if epoch < 21:
                # 进行warmup，逐渐增加学习率
                warm_iteration = 30 * 213
                lr = 0.00035 * iteration / warm_iteration
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
            if True: #正则化参数warmup
                new_weight_decay = some_function(epoch, 0.5)
                for param_group in optimizer.param_groups:
                    param_group['weight_decay'] = new_weight_decay

        optimizer.zero_grad()

        loss, metric = model(data, labels,
                             cam_ids=cam_ids,
                             epoch=epoch)


        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
        optimizer.step()

        return metric

    return Engine(_process_func)


def create_eval_engine(model, non_blocking=False):
    device = torch.device("cuda", torch.cuda.current_device())

    def _process_func(engine, batch):
        model.eval()

        data, labels, cam_ids, img_paths = batch[:4]

        data = data.to(device, non_blocking=non_blocking)

        with no_grad():
            feat = model(data, cam_ids=cam_ids.to(device, non_blocking=non_blocking))

        return feat.data.float().cpu(), labels, cam_ids, np.array(img_paths)

    engine = Engine(_process_func)

    @engine.on(Events.EPOCH_STARTED)
    def clear_data(engine):
        # feat list
        if not hasattr(engine.state, "feat_list"):
            setattr(engine.state, "feat_list", [])
        else:
            engine.state.feat_list.clear()

        # id_list
        if not hasattr(engine.state, "id_list"):
            setattr(engine.state, "id_list", [])
        else:
            engine.state.id_list.clear()

        # cam list
        if not hasattr(engine.state, "cam_list"):
            setattr(engine.state, "cam_list", [])
        else:
            engine.state.cam_list.clear()

        # img path list
        if not hasattr(engine.state, "img_path_list"):
            setattr(engine.state, "img_path_list", [])
        else:
            engine.state.img_path_list.clear()

    @engine.on(Events.ITERATION_COMPLETED)
    def store_data(engine):
        engine.state.feat_list.append(engine.state.output[0])
        engine.state.id_list.append(engine.state.output[1])
        engine.state.cam_list.append(engine.state.output[2])
        engine.state.img_path_list.append(engine.state.output[3])

    return engine
