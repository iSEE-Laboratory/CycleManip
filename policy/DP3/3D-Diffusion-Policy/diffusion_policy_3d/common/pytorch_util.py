from typing import Dict, Callable, List
import collections
import torch
import torch.nn as nn

def dict_apply(x: Dict[str, torch.Tensor], func: Callable[[torch.Tensor], torch.Tensor]) -> Dict[str, torch.Tensor]:
    result = dict()
    for key, value in x.items():
        if isinstance(value, dict):
            result[key] = dict_apply(value, func)
        elif isinstance(value, list):
            # 且里面的元素是str
            result[key] = value
        else:
            result[key] = func(value)
    return result

# def dict_apply(x: Dict[str, torch.Tensor], func: Callable[[torch.Tensor], torch.Tensor]) -> Dict[str, torch.Tensor]:
#     result = dict()
#     for key, value in x.items():
#         if isinstance(value, dict):
#             result[key] = dict_apply(value, func)

#         elif isinstance(value, list):
#             # 若 list 内元素为字符串，则不处理
#             if len(value) > 0 and all(isinstance(v, str) for v in value):
#                 result[key] = value
#             else:
#                 # 否则对 list 内 tensor 同样进行 func 转换
#                 result[key] = [func(v) for v in value]

#         else:
#             result[key] = func(value)

#     return result


def pad_remaining_dims(x, target):
    assert x.shape == target.shape[:len(x.shape)]
    return x.reshape(x.shape + (1, ) * (len(target.shape) - len(x.shape)))


def dict_apply_split(
    x: Dict[str, torch.Tensor],
    split_func: Callable[[torch.Tensor], Dict[str, torch.Tensor]],
) -> Dict[str, torch.Tensor]:
    results = collections.defaultdict(dict)
    for key, value in x.items():
        result = split_func(value)
        for k, v in result.items():
            results[k][key] = v
    return results


def dict_apply_reduce(
    x: List[Dict[str, torch.Tensor]],
    reduce_func: Callable[[List[torch.Tensor]], torch.Tensor],
) -> Dict[str, torch.Tensor]:
    result = dict()
    for key in x[0].keys():
        result[key] = reduce_func([x_[key] for x_ in x])
    return result


def optimizer_to(optimizer, device):
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device=device)
    return optimizer
