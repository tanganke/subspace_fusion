from typing import Dict, List

import torch
from torch import Tensor, nn


def state_dicts_check_keys(state_dicts: List[Dict[str, Tensor]]):
    """
    Checks that the state dictionaries have the same keys.

    Args:
        state_dicts (List[Dict[str, Tensor]]): A list of dictionaries containing the state of PyTorch models.

    Raises:
        ValueError: If the state dictionaries have different keys.
    """
    # Get the keys of the first state dictionary in the list
    keys = set(state_dicts[0].keys())
    # Check that all the state dictionaries have the same keys
    for state_dict in state_dicts:
        assert keys == set(state_dict.keys()), "keys of state_dicts are not equal"


def num_params_of_state_dict(state_dict: Dict[str, Tensor]):
    """
    Returns the number of parameters in a state dict.

    Args:
        state_dict (Dict[str, Tensor]): The state dict to count the number of parameters in.

    Returns:
        int: The number of parameters in the state dict.
    """
    return sum([state_dict[key].numel() for key in state_dict])


def state_dict_flatten(state_dict: Dict[str, Tensor]):
    """
    Flattens a state dict.

    Args:
        state_dict (Dict[str, Tensor]): The state dict to be flattened.

    Returns:
        Tensor: The flattened state dict.
    """
    flattened_state_dict = []
    for key in state_dict:
        flattened_state_dict.append(state_dict[key].flatten())
    return torch.cat(flattened_state_dict)


def state_dict_avg(state_dicts: List[Dict[str, Tensor]]):
    """
    Returns the average of a list of state dicts.

    Args:
        state_dicts (List[Dict[str, Tensor]]): The list of state dicts to average.

    Returns:
        Dict: The average of the state dicts.
    """
    assert len(state_dicts) > 0, "The number of state_dicts must be greater than 0"
    assert all([len(state_dicts[0]) == len(state_dict) for state_dict in state_dicts]), "All state_dicts must have the same number of keys"

    avg_state_dict = {}
    for key in state_dicts[0]:
        avg_state_dict[key] = torch.zeros_like(state_dicts[0][key])
        for state_dict in state_dicts:
            avg_state_dict[key] += state_dict[key]
        avg_state_dict[key] /= len(state_dicts)
    return avg_state_dict


def state_dict_sub(a: Dict, b: Dict, strict: bool = True):
    """
    Returns the difference between two state dicts.

    Args:
        a (Dict): The first state dict.
        b (Dict): The second state dict.
        strict (bool): Whether to check if the keys of the two state dicts are the same.

    Returns:
        Dict: The difference between the two state dicts.
    """
    if strict:
        assert set(a.keys()) == set(b.keys())

    diff = {}
    for k in a:
        if k in b:
            diff[k] = a[k] - b[k]
    return diff


def state_dict_add(a: Dict, b: Dict, strict: bool = True):
    """
    Returns the sum of two state dicts.

    Args:
        a (Dict): The first state dict.
        b (Dict): The second state dict.
        strict (bool): Whether to check if the keys of the two state dicts are the same.

    Returns:
        Dict: The sum of the two state dicts.
    """
    if strict:
        assert set(a.keys()) == set(b.keys())

    diff = {}
    for k in a:
        if k in b:
            diff[k] = a[k] + b[k]
    return diff


def state_dict_mul(state_dict: Dict, scalar: float):
    """
    Returns the product of a state dict and a scalar.

    Args:
        state_dict (Dict): The state dict to be multiplied.
        scalar (float): The scalar to multiply the state dict with.

    Returns:
        Dict: The product of the state dict and the scalar.
    """
    diff = {}
    for k in state_dict:
        diff[k] = scalar * state_dict[k]
    return diff


def state_dict_power(state_dict: Dict[str, Tensor], p: float):
    """
    Returns the power of a state dict.

    Args:
        state_dict (Dict[str, Tensor]): The state dict to be powered.
        p (float): The power to raise the state dict to.

    Returns:
        Dict[str, Tensor]: The powered state dict.
    """
    powered_state_dict = {}
    for key in state_dict:
        powered_state_dict[key] = state_dict[key] ** p
    return powered_state_dict


def state_dict_interpolation(state_dicts: List[Dict[str, Tensor]], scalars: List[float]):
    """
    Interpolates between a list of state dicts using a list of scalars.

    Args:
        state_dicts (List[Dict[str, Tensor]]): The list of state dicts to interpolate between.
        scalars (List[float]): The list of scalars to use for interpolation.

    Returns:
        Dict: The interpolated state dict.
    """
    assert len(state_dicts) == len(scalars), "The number of state_dicts and scalars must be the same"
    assert len(state_dicts) > 0, "The number of state_dicts must be greater than 0"
    assert all([len(state_dicts[0]) == len(state_dict) for state_dict in state_dicts]), "All state_dicts must have the same number of keys"

    interpolated_state_dict = {}
    for key in state_dicts[0]:
        interpolated_state_dict[key] = torch.zeros_like(state_dicts[0][key])
        for state_dict, scalar in zip(state_dicts, scalars):
            interpolated_state_dict[key] += scalar * state_dict[key]
    return interpolated_state_dict


def state_dict_sum(state_dicts: List[Dict[str, Tensor]]):
    """
    Returns the sum of a list of state dicts.

    Args:
        state_dicts (List[Dict[str, Tensor]]): The list of state dicts to sum.

    Returns:
        Dict: The sum of the state dicts.
    """
    assert len(state_dicts) > 0, "The number of state_dicts must be greater than 0"
    assert all([len(state_dicts[0]) == len(state_dict) for state_dict in state_dicts]), "All state_dicts must have the same number of keys"

    sum_state_dict = {}
    for key in state_dicts[0]:
        sum_state_dict[key] = torch.zeros_like(state_dicts[0][key])
        for state_dict in state_dicts:
            sum_state_dict[key] += state_dict[key]
    return sum_state_dict


def state_dict_weighted_sum(state_dicts: List[Dict[str, Tensor]], weights: List[float]):
    """
    Returns the weighted sum of a list of state dicts.

    Args:
        state_dicts (List[Dict[str, Tensor]]): The list of state dicts to interpolate between.
        weights (List[float]): The list of weights to use for the weighted sum.

    Returns:
        Dict: The weighted sum of the state dicts.
    """
    assert len(state_dicts) == len(weights), "The number of state_dicts and weights must be the same"
    assert len(state_dicts) > 0, "The number of state_dicts must be greater than 0"
    assert all([len(state_dicts[0]) == len(state_dict) for state_dict in state_dicts]), "All state_dicts must have the same number of keys"

    weighted_sum_state_dict = {}
    for key in state_dicts[0]:
        weighted_sum_state_dict[key] = torch.zeros_like(state_dicts[0][key])
        for state_dict, weight in zip(state_dicts, weights):
            weighted_sum_state_dict[key] += weight * state_dict[key]
    return weighted_sum_state_dict
