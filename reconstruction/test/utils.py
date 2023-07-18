import hdf5storage
import torch
import torch.nn as nn
import numpy as np
#from fvcore.nn import FlopCountAnalysis

def save_matv73(mat_name, var_name, var):
    hdf5storage.savemat(mat_name, {var_name: var}, format='7.3', store_python_metadata=True)

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class Loss_MRAE(nn.Module):
    def __init__(self):
        super(Loss_MRAE, self).__init__()

    def forward(self, outputs, label):
        assert outputs.shape == label.shape
        label = torch.add(label,0.0001)
        error = torch.abs(outputs - label) / label
        mrae = torch.mean(error.reshape(-1))
        return mrae

class Loss_RMSE(nn.Module):
    def __init__(self):
        super(Loss_RMSE, self).__init__()

    def forward(self, outputs, label):
        assert outputs.shape == label.shape
        error = outputs-label
        sqrt_error = torch.pow(error,2)
        rmse = torch.sqrt(torch.mean(sqrt_error.reshape(-1)))
        return rmse

class Loss_SID(nn.Module):
    def __init__(self):
        super(Loss_RMSE, self).__init__()

    def forward(self, outputs, label):
        assert outputs.shape == label.shape
        error = outputs-label
        sqrt_error = torch.pow(error,2)
        rmse = torch.sqrt(torch.mean(sqrt_error.view(-1)))
        return rmse

class Loss_SAM(nn.Module):
    def __init__(self):
        super(Loss_SAM, self).__init__()

    def forward(self, tensor_pred, tensor_gt):
        assert tensor_pred.shape == tensor_gt.shape
        dot = torch.sum(tensor_pred * tensor_gt, dim=1).view(-1)
        # norm calculations
        image = tensor_pred.reshape(-1, tensor_pred.shape[1])
        norm_original = torch.norm(image, p=2, dim=1)

        target = tensor_gt.reshape(-1, tensor_gt.shape[1])
        norm_reconstructed = torch.norm(target, p=2, dim=1)

        norm_product = (norm_original.mul(norm_reconstructed)).pow(-1)
        argument = dot.mul(norm_product)
        # for avoiding arccos(1)
        acos = torch.acos(torch.clamp(argument, -1 + 1e-7, 1 - 1e-7))
        loss = torch.mean(acos)

        if torch.isnan(loss):
            raise ValueError(
                f"Loss is NaN value. Consecutive values - dot: {dot}, \
            norm original: {norm_original}, norm reconstructed: {norm_reconstructed}, \
            norm product: {norm_product}, argument: {argument}, acos: {acos}, \
            loss: {loss}, input: {tensor_pred}, output: {target}"
            )
        return loss

class Loss_PSNR(nn.Module):
    def __init__(self):
        super(Loss_PSNR, self).__init__()

    def forward(self, im_true, im_fake, data_range=1):
        N = im_true.size()[0]
        C = im_true.size()[1]
        H = im_true.size()[2]
        W = im_true.size()[3]
        Itrue = im_true.clamp(0., 1.).mul_(data_range).resize_(N, C * H * W)
        Ifake = im_fake.clamp(0., 1.).mul_(data_range).resize_(N, C * H * W)
        mse = nn.MSELoss(reduce=False)
        err = mse(Itrue, Ifake).sum(dim=1, keepdim=True).div_(C * H * W)
        psnr = 10. * torch.log((data_range ** 2) / err) / np.log(10.)
        return torch.mean(psnr)

def my_summary(test_model, H = 512, W = 512, C = 68, N = 1):
    model = test_model.cuda()
    print(model)
    inputs = torch.randn((N, C, H, W)).cuda()
    #flops = FlopCountAnalysis(model,inputs)
    n_param = sum([p.nelement() for p in model.parameters()])
    print(f'GMac:{flops.total()/(1024*1024*1024)}')
    print(f'Params:{n_param}')
