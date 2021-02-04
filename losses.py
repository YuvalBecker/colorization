import torch


#def custom_loss_binary(y_pred1, y_pred2):
#    return torch.mean(torch.abs((y_pred1) * y_pred2 + (1 - y_pred2) * (1 - y_pred1)) )
#
#def custom_loss_binary(y_pred1, y_pred2):
#    return torch.mean(1/torch.abs((y_pred1 - y_pred2)) )


def custom_loss_binary(y_pred1, y_pred2):
    return torch.mean(torch.abs(y_pred1) * torch.abs(y_pred2) + 1/y_pred1**2 +1/y_pred2**2 )


def total_varation(output):
    tv_loss = torch.sum(torch.abs(output[:, :, :, :-1] - output[:, :, :, 1:])) \
              + torch.sum(torch.abs(output[:, :, :-1, :] - output[:, :, 1:, :]))
    return tv_loss
