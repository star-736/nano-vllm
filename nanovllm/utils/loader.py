import os
from glob import glob
import torch
from torch import nn
from safetensors import safe_open

def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor):
    """默认的权重加载方法"""
    param.data.copy_(loaded_weight)


def load_model(model: nn.Module, path: str):
    """加载layers里不同类定义的weight_loader方法"""
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {}) # 获取模型的packed_modules_mapping属性，用于参数名称映射与分片
    for file in glob(os.path.join(path, "*.safetensors")):
        with safe_open(file, "pt", "cpu") as f:
            for weight_name in f.keys():
                # print(f"{weight_name}, {f.get_tensor(weight_name).shape}")
                for k in packed_modules_mapping: # for-else循环
                    if k in weight_name:
                        v, shard_id = packed_modules_mapping[k]
                        param_name = weight_name.replace(k, v)
                        param = model.get_parameter(param_name)
                        weight_loader = getattr(param, "weight_loader")
                        weight_loader(param, f.get_tensor(weight_name), shard_id)
                        break
                else:
                    param = model.get_parameter(weight_name) # 会去对应类里找nn.Parameter这一行，维度已在初始化各模块时按照tp_size设定好
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    weight_loader(param, f.get_tensor(weight_name)) # 加载对应组件的预训练权重
