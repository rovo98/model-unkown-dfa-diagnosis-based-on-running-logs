# using techniques like over-sampling and under-sampling to deal with the imbalanced raw data.
# rovo98
# since 2019.12.24

import random
import copy


def over_sampling(given_list, target_size):
    """Do over sampling on the given list.

    :type given_list: list
    :type target_size: int

    :param given_list: <list> A list contains any type of elements.
    :param target_size: the target size of the new list after doing over sampling.
    :return: A list of new elements after doing over sampling with the length of `target_size`
    """
    # base checking
    if len(given_list) > target_size:
        raise Exception('the size of the given list is greater than the target size.')

    original_size = len(given_list)
    bias = target_size - original_size

    # FIXME: temporary implementation
    # avoiding extending to many samples
    if bias > original_size * 0.5:
        bias = bias // 2
        print('>> updated: ', original_size, ' -> ', original_size + bias)

    result = copy.copy(given_list)

    if bias < original_size:
        result.extend(random.sample(given_list, k=bias))
    else:
        epoch = bias // original_size
        bias = bias - (original_size * epoch)
        for _ in range(epoch):
            result.extend(copy.copy(given_list))
        if bias > 0:
            result.extend(random.sample(given_list, k=bias))
    return result


def under_sampling(given_list, target_size):
    """Do under sampling on the given list, and returns the result.

    :type given_list: list
    :type target_size: int

    :param given_list: <list> A list containing any type of elements.
    :param target_size: <int> the new size of the result list after doing under sampling.
    :return: A result list contains elements after doing under sampling operation.
    """
    # basic validation is needed
    original_size = len(given_list)
    if original_size <= target_size:
        raise Exception('len of the given list is smaller than the target size!')

    # FIXME: temporary implementation
    # reducing more sampling when there is a extra big dataset.
    if original_size - target_size > original_size * 0.5:
        target_size = int(original_size * 0.5)
        print('>> updated: ', original_size, ' -> ', target_size)

    result = []
    result.extend(random.sample(given_list, target_size))
    return result
