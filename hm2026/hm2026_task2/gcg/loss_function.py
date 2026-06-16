# Some useful libraries, feel free to import any others you need.
import torch.nn.functional as F

from torch import Tensor

#todo Your job is to compute the loss used by GCG, which is used in line 538 of algorithm.py 

def compute_loss(shift_logits: Tensor, shift_labels: Tensor) -> Tensor:
    loss = ...

    # ===================== TODO: the GCG objective =====================
    # GCG works by making the model's target completion as likely as possible.
    # Concretely, it MINIMIZES the negative log-likelihood of the target string
    # given the prompt + adversarial suffix (Eq. (2)/(3) in the paper):
    #
    #     L(x) = - log p(target | prompt + suffix)
    #          = - sum_i  log p( target_i | everything before it )
    #
    # For each target position this is exactly the cross-entropy between the
    # model's predicted next-token distribution (`shift_logits`) and the true
    # next token (`shift_labels`). The logits/labels have already been shifted
    # by the caller so that they line up position-by-position.
    #
    # IMPORTANT: return the PER-TOKEN loss (do NOT reduce it here). The callers
    # in algorithm.py reshape it to (batch, num_target_ids) and average over the
    # target tokens themselves -- so use reduction="none".
    #
    # Hint: F.cross_entropy(logits_2d, labels_1d, reduction="none"), where you
    #       flatten the leading dims:
    #         logits -> (batch * num_target_ids, vocab_size)
    #         labels -> (batch * num_target_ids,)
    loss = ...  # TODO: your code here
    # ===================================================================
    
    return loss
