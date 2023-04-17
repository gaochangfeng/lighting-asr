#!/usr/bin/env python
import argparse
import os
import torch


def model_average(path, ids = "best", num=10):
    '''
        平均文件夹下"last-step-{epoch:02d}-{global_step}"和"best-val-{valid_loss:.6f}-{epoch:02d}"
        的模型。
    '''
    new_model_list = []
    for model in os.listdir(path):
        if os.path.splitext(model)[-1] == '.ckpt':
            new_model_list.append(model)
    new_model_list.sort(reverse  = ids=="last")
    choose = new_model_list[:num]

        # sum
    avg = None
    for model in choose:
        model_path = os.path.join(path,model) 
        states = torch.load(model_path, map_location=torch.device("cpu"))["state_dict"]
        if avg is None:
            avg = states
        else:
            for k in avg.keys():
                avg[k] += states[k]

        # average
    for k in avg.keys():
        if avg[k] is None:
            continue
        if avg[k].dtype in (torch.int32, torch.int64, torch.uint8) :
            avg[k] //= num
        else:
            avg[k] /= num
    return avg, choose

