# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch

# pylint: disable=no-member,not-callable


def modified_euler_integrate(func, init_state, times):
    """Modified Euler method for numerical integration of initial value problems"""
    x = [init_state]
    h = times[1] - times[0]
    for t2, t1 in zip(times[1:], times[:-1]):
        f1 = func(t1, x[-1])
        f2 = func(t2, x[-1] + h * f1)
        x.append(x[-1] + 0.5 * h * (f1 + f2))
    return torch.stack(x)


def modified_euler_while_body(func, x0, t1, t2):
    h = t2 - t1
    f1 = func(t1, x0)
    f2 = func(t2, x0 + h * f1)
    x1 = x0 + 0.5 * h * (f1 + f2)
    return x1


def modified_euler_while(func, init_state, times):
    """Modified Euler method for numerical integration of initial value problems, using a while loop"""
    x0 = init_state  # [init_state]
    T = torch.tensor([len(times)]).type(torch.LongTensor)
    results = [x0]

    i = torch.tensor([1]).type(torch.LongTensor)
    while (i < T)[0]:
        x1 = modified_euler_while_body(func, x0, times[i[0] - 1], times[i[0]])
        x0 = x1
        results.append(x1)
        i += 1

    return torch.stack(results)
