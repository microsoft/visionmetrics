import torch


def targets_to_mat(targets, n_class):
    if len(targets.shape) == 1:
        target_mat = torch.zeros((len(targets), n_class), dtype=int)
        for i, t in enumerate(targets):
            target_mat[i, t] = 1
    else:
        target_mat = targets

    return target_mat
