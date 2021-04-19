import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

def _dice_loss(predict, target):

	smooth = 1e-5

	y_true_f = target.contiguous().view(target.shape[0], -1)
	y_pred_f = predict.contiguous().view(predict.shape[0], -1)
	intersection = torch.sum(torch.mul(y_pred_f, y_true_f), dim=1)
	union = torch.sum(y_pred_f, dim=1) + torch.sum(y_true_f, dim=1) + smooth
	dice_score = (2.0 * intersection / union)

	dice_loss = 1 - dice_score

	return dice_loss


class Dice_Loss(nn.Module):
	def __init__(self):
		super(Dice_Loss, self).__init__()

	def forward(self, predicts, target):

		preds = torch.softmax(predicts, dim=1)
		dice_loss0 = _dice_loss(preds[:, 0, :, :], 1 - target)
		dice_loss1 = _dice_loss(preds[:, 1, :, :], target)
		loss_D = (dice_loss0.mean() + dice_loss1.mean())/2.0

		return loss_D


class Task_Interaction_Loss(nn.Module):
    
    def __init__(self):
        super(Task_Interaction_Loss, self).__init__()

    def forward(self, cls_predict, seg_predict, target):
        
        b,c = cls_predict.shape
        h, w = seg_predict.shape[2], seg_predict.shape[3]
        
        target = target.view(b,1)
        target = torch.zeros(b,c).cuda().scatter_(1,target,1)
        target_new = torch.zeros(b,c-1).cuda()
        cls_pred = Variable(torch.zeros(b,c-1)).cuda()
        seg_pred = Variable(torch.zeros(b,c-1)).cuda()
    
        target_new[:,0] = target[:,0]
        target_new[:,1] = target[:,1] + target[:,2] 
        
        cls_pred[:,0] = cls_predict[:,0]
        cls_pred[:,1] = cls_predict[:,1]  + cls_predict[:,2]  
       
        # Log Sum Exp
        seg_pred = torch.logsumexp(seg_predict, dim=(2,3))/(h*w)
    
        #JS
        seg_cls_kl = F.kl_div(torch.log_softmax(cls_pred, dim=-1), torch.softmax(seg_pred, dim=-1), reduction='none')
        cls_seg_kl = F.kl_div(torch.log_softmax(seg_pred, dim=-1), torch.softmax(cls_pred, dim=-1), reduction='none')
        
        seg_cls_kl = seg_cls_kl.sum(-1)
        cls_seg_kl = cls_seg_kl.sum(-1)
        
        indicator1 = (cls_pred[:,0]>cls_pred[:,1]) == (target_new[:,0]>target_new[:,1])
        indicator2 = (seg_pred[:,0]>seg_pred[:,1]) == (target_new[:,0]>target_new[:,1])
    
        return (cls_seg_kl*indicator1 + seg_cls_kl*indicator2).sum()/2./b



    
    