from torch import nn

class MLMLoss(nn.Module):
    """
    Masked Language Modeling loss.
    """
    def __init__(self, mse_based=False):
        super(MLMLoss, self).__init__()
        self.mse_based = mse_based
        if self.mse_based:
            self.loss_fn = nn.MSELoss(reduction='none')
        else:
            self.loss_fn = nn.CrossEntropyLoss(reduction='none')

    def forward(self, predictions, targets, mask):
        if self.mse_based:
            predictions = predictions.squeeze(-1)
        else:
            predictions = predictions.permute(0, 2, 1) # (batch_size, vocab_size, seq_len)
            targets = targets.long()
            
        masked_loss = self.loss_fn(predictions, targets)
        masked_loss = masked_loss * mask.float() 
        return masked_loss.sum() / mask.sum()