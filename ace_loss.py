import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

def one_hot(tensors, num_classes):
    onehot = []
    tensors = tensors.cuda()
    for tensor in tensors:
        tensor = tensor.unsqueeze(1)
        t = torch.zeros(tensor.shape[0], num_classes).cuda().scatter_(1, tensor, 1)
        onehot.append(t)
    onehot = torch.stack(onehot)
    return onehot

class ACELoss(nn.Module):
    def __init__(self, alpha=0, size_average=False):
        super(ACELoss, self).__init__()
        self.alpha = alpha
        self.size_average = size_average

    def forward(self, logits, targets, input_lengths, target_lengths):
        T_, B, C = logits.size()

        tagets_split = list(torch.split(targets, target_lengths.tolist()))
        targets_padded = torch.nn.utils.rnn.pad_sequence(tagets_split, batch_first=True, padding_value=0)
        targets_padded = one_hot(targets_padded.long(), num_classes=C)

        targets_padded = (targets_padded * (1-self.alpha)) + (self.alpha/C)
        targets_padded = torch.sum(targets_padded, 1).float().cuda()
        targets_padded[:,0] = T_ - target_lengths

        probs = torch.softmax(logits, dim=2)
        probs = torch.sum(probs, 0)
        probs = probs / T_
        targets_padded = targets_padded / T_

        #targets_padded = F.normalize(targets_padded, p=1, dim=1)
        #loss = F.kl_div(torch.log(probs), targets_padded, reduction='sum')

        #print(-torch.sum(torch.log(probs[0]) * targets_padded[0])) , (-torch.sum(torch.log(probs[1:]) * targets_padded[1:]))
        loss = -torch.sum(torch.log(probs) * targets_padded) / B

        return loss


if __name__ == '__main__':
    ace = ACELoss().cuda()

    pred_np = np.array([[0.5, 0.4, 0.1], [0.3, 0.1, 0.6], [0.7, 0.2, 0.1], [0.3, 0.5, 0.2]])[:, None]
    pred_np = np.log(np.tile(pred_np, (1,2,1)))
    # (U)
    token_np = np.array([2, 2, 1, 1, 2])

    pred = torch.FloatTensor(pred_np).cuda()
    token = torch.IntTensor(token_np).cuda()
    sizes = torch.IntTensor(np.array([4, 4])).cuda()
    target_sizes = torch.IntTensor(np.array([3, 2])).cuda()

    loss = ace(pred, token, sizes, target_sizes)
    print(loss)
