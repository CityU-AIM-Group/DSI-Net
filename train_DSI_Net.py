import torch
import torch.nn as nn
import numpy as np
import torch.backends.cudnn as cudnn
import os
from sklearn import metrics
from sklearn.metrics import accuracy_score
from net.models import DSI_Net
from net.loss import Task_Interaction_Loss, Dice_Loss
from dataset.my_datasets import CADCAPDataset
from torch.utils import data
from apex import amp
from utils.logger import print_f
import time
import config
import argparse
from visualization.utils import show_seg_results, draw_curves

#https://drive.google.com/file/d/12RjjEKM4nXtskHSJkWMdJ7S5PeaVFow3/view?usp=sharing
model_urls = {'deeplabv3plus_xception': 'data/pre_model/deeplabv3plus_xception_VOC2012_epoch46_all.pth'}

parser = argparse.ArgumentParser()
parser.add_argument('--image_list', default='/home/meiluzhu2/data/WCE/WCE_Dataset_larger_Fold1.pkl', type=str, help='image list pkl')
parser.add_argument('--gpus', default='7', type=str, help='gpus')
parser.add_argument('--K', default=100, type=int, help='seed number')
parser.add_argument('--alpha', default=0.05, type=float, help='the weight of interaction loss')
args = parser.parse_args()

def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1 - float(iter) / max_iter) ** (power))

def adjust_learning_rate(optimizer, i_iter):
    lr = lr_poly(config.LEARNING_RATE, i_iter, config.STEPS, config.POWER)
    optimizer.param_groups[0]['lr'] = lr
    return lr

def test(valloader, model, epoch, path = None, verbose = False):
    # valiadation
    #cls
    pro_score_crop = []
    label_val_crop = []
    
    #refine seg
    seg_dice = []
    seg_sen = []
    seg_spe = []
    seg_acc = []
    seg_jac_score = []     
    
    for index, batch in enumerate(valloader):
        data, masks, label, name = batch
        data = data.cuda() 
        label = label.cuda()
        mask = masks[0].data.numpy()                   
        val_mask = np.int64(mask > 0)        
        
        model.eval()
        with torch.no_grad():
            pred_seg_coarse, pred_seg_fine, pred_cls = model(data) 
            
        #cls
        pro_score_crop.append(torch.softmax(pred_cls[0], dim=0).cpu().data.numpy())
        label_val_crop.append(label[0].cpu().data.numpy())

        #seg
        y_true_f = val_mask.reshape(val_mask.shape[0]*val_mask.shape[1], order='F')                 
        if np.sum(y_true_f) != 0 and label[0].cpu().data.numpy() != 0:
            pred_seg = torch.softmax(pred_seg_fine, dim=1).cpu().data.numpy()
            pred_arg = np.argmax(pred_seg[0], axis=0) 
            y_pred_f = pred_arg.reshape(pred_arg.shape[0]*pred_arg.shape[1], order='F')  
            intersection = np.float(np.sum(y_true_f * y_pred_f))
            seg_dice.append((2. * intersection) / (np.sum(y_true_f) + np.sum(y_pred_f)))
            seg_sen.append(intersection / np.sum(y_true_f))
            intersection0 = np.float(np.sum((1 - y_true_f) * (1 - y_pred_f)))
            seg_spe.append(intersection0 / np.sum(1 - y_true_f))
            seg_acc.append(accuracy_score(y_true_f, y_pred_f))
            seg_jac_score.append(intersection / (np.sum(y_true_f) + np.sum(y_pred_f) - intersection))
            
            if verbose == config.VERBOSE and epoch == config.EPOCH-1:
                show_seg_results(data[0].cpu().data.numpy().transpose(1, 2, 0), mask, pred_arg, path, name[0])
    #cls
    pro_score_crop = np.array(pro_score_crop)
    label_val_crop = np.array(label_val_crop)
    binary_score = np.eye(3)[np.argmax(np.array(pro_score_crop), axis=-1)]
    label_val = np.eye(3)[np.int64(np.array(label_val_crop))]
    preds = np.argmax(np.array(pro_score_crop), axis=-1)
    CK = metrics.cohen_kappa_score(label_val_crop, preds)     
    OA = metrics.accuracy_score(label_val_crop, preds)
    EREC = metrics.recall_score(label_val, binary_score, average=None)   

    result = {}
    result['seg'] = [np.array(seg_acc), np.array(seg_dice), np.array(seg_sen), np.array(seg_spe), np.array(seg_jac_score)]
    result['cls'] = [CK, OA, EREC]    
    return result


def main():
    """Create the network and start the training."""
   
    cudnn.enabled = True
    cudnn.benchmark = True    
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    
    ############# Create mask-guided classification network.
    model = DSI_Net(config, K = args.K)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay =config.WEIGHT_DECAY)
    model.cuda()
    if config.FP16 is True:
        model, optimizer = amp.initialize(model, optimizer, opt_level="O1")
    model = torch.nn.DataParallel(model)
    
    ############# Load pretrained weights
    pretrained_dict = torch.load(model_urls['deeplabv3plus_xception'])
    net_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if (k in net_dict) and (v.shape == net_dict[k].shape)}
    net_dict.update(pretrained_dict)
    model.load_state_dict(net_dict)
    print(len(net_dict))
    print(len(pretrained_dict))
    model.train()
    model.float()
    ce_loss = nn.CrossEntropyLoss()
    dice_loss = Dice_Loss()
    task_interaction_loss = Task_Interaction_Loss()
    
    ############# Load training and validation data
    trainloader = data.DataLoader(CADCAPDataset(config.DATA_ROOT, args.image_list, config.SIZE, data_type='train', mode = 'train'), batch_size=config.BATCH_SIZE, shuffle=True, 
                                  num_workers=config.NUM_WORKERS, pin_memory=True, drop_last = config.DROP_LAST)
    testloader = data.DataLoader(CADCAPDataset(config.DATA_ROOT, args.image_list,config.SIZE, data_type = 'test', mode='test'), 
                                batch_size=1, shuffle=False, num_workers=config.NUM_WORKERS, pin_memory=True)
    train_testloader = data.DataLoader(CADCAPDataset(config.DATA_ROOT, args.image_list,config.SIZE, data_type = 'train', mode = 'test'), 
                                batch_size=1, shuffle=False, num_workers=config.NUM_WORKERS, pin_memory=True)    
    
    
    if not os.path.isdir(config.SAVE_PATH):
        os.mkdir(config.SAVE_PATH)
    if not os.path.isdir(config.SAVE_PATH+'Seg_results/'):
        os.mkdir(config.SAVE_PATH+'Seg_results/')
    if not os.path.isdir(config.LOG_PATH):
        os.mkdir(config.LOG_PATH)    
    
    f_path = config.LOG_PATH + 'training_output.log'
    logfile = open(f_path, 'a')
    
    print_f(os.getcwd(), f=logfile)
    print_f('Device: {}'.format(args.gpus), f=logfile)
    print_f('==={}==='.format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())), f=logfile)
    print_f('===Setting===', f=logfile)
    print_f('    Data_list: {}'.format(args.image_list), f=logfile)
    print_f('    K: {}'.format(args.K), f=logfile)
    print_f('    Lost_weight: {}'.format(args.alpha), f=logfile)
    print_f('    LR: {}'.format(config.LEARNING_RATE), f=logfile)  

    OA_bulk_train = []
    CK_bulk_train = []
    DI_bulk_train = []
    JA_bulk_train = []
    SE_bulk_train = []

    OA_bulk_test = []
    CK_bulk_test = []
    DI_bulk_test = []
    JA_bulk_test = []
    SE_bulk_test = []    
    
        
    for epoch in range(config.EPOCH):   
        #cls
        cls_train_loss = []
        seg_train_loss = []
        train_inter_loss = []              
        ############# Start the training
        for i_iter, batch in enumerate(trainloader):
            step = (config.TRAIN_NUM/config.BATCH_SIZE)*epoch+i_iter    
            images, masks, labels, name = batch
            images = images.cuda()
            labels = labels.cuda().long()
            masks = masks.cuda().squeeze(1)
            optimizer.zero_grad()
            lr = adjust_learning_rate(optimizer, step)
            model.train()
            preds_seg_coarse, preds_seg_fine, preds_cls = model(images)
            cls_loss = ce_loss(preds_cls, labels)            
            seg_loss_fine = dice_loss(preds_seg_fine, masks)
            seg_loss_coarse = dice_loss(preds_seg_coarse, masks) 
            inter_loss  = task_interaction_loss(preds_cls, preds_seg_fine, labels)
            loss = cls_loss + seg_loss_fine + seg_loss_coarse + args.alpha * inter_loss
                                   
            if config.FP16 is True:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            optimizer.step()
            #cls
            cls_train_loss.append(cls_loss.cpu().data.numpy())  
            seg_train_loss.append(seg_loss_fine.cpu().data.numpy())
            train_inter_loss.append(inter_loss.cpu().data.numpy())            
        
        ############ train log
        line = "Train-Epoch [%d/%d] [All]: Seg_loss = %.6f, Class_loss = %.6f, Inter_loss = %.6f, LR = %0.9f\n" % (epoch, config.EPOCH, np.nanmean(seg_train_loss), np.nanmean(cls_train_loss), np.nanmean(train_inter_loss), lr)
        print_f(line, f=logfile)                                                  
        
        result = test(train_testloader, model, epoch, verbose=False)
        #cls
        [CK, OA, EREC] = result['cls']
        OA_bulk_train.append(OA)
        CK_bulk_train.append(CK)
        
        # seg               
        [AC, DI, SE, SP, JA] = result['seg']        
        JA_bulk_train.append(np.nanmean(JA))  
        DI_bulk_train.append(np.nanmean(DI))
        SE_bulk_train.append(np.nanmean(SE))        
        
        ############# Start the test
        result = test(testloader, model, epoch, config.SAVE_PATH+'Seg_results/' , verbose = config.VERBOSE)
        #cls
        [CK, OA, EREC] = result['cls']
        line = "Test -Epoch [%d/%d] [Cls]: CK-Score = %f, OA = %f, Rec-N = %f, Rec-V = %f, Rec-I=%f \n" % (epoch, config.EPOCH, CK, OA, EREC[0],EREC[1],EREC[2] )
        print_f(line, f=logfile)
        OA_bulk_test.append(OA)
        CK_bulk_test.append(CK)
                                            
        # seg               
        [AC, DI, SE, SP, JA] = result['seg'] 
        line = "Test -Epoch [%d/%d] [Seg]: AC = %f, DI = %f, SE = %f, SP = %f, JA = %f \n" % (epoch, config.EPOCH, np.nanmean(AC), np.nanmean(DI), np.nanmean(SE), np.nanmean(SP), np.nanmean(JA))
        print_f(line, f=logfile)
        
        JA_bulk_test.append(np.nanmean(JA))  
        DI_bulk_test.append(np.nanmean(DI))
        SE_bulk_test.append(np.nanmean(SE))
        
        ############# Plot val curve
        filename = os.path.join(config.LOG_PATH, 'cls_curves.png')
        data_list = [OA_bulk_train, OA_bulk_test, CK_bulk_train, CK_bulk_test]
        label_list = ['OA_train','OA_test','CK_train','CK_test']
        draw_curves(data_list = data_list, label_list = label_list, color_list = config.COLOR[0:4], filename = filename)
        filename = os.path.join(config.LOG_PATH, 'seg_curves.png')
        data_list = [JA_bulk_train, JA_bulk_test, DI_bulk_train, DI_bulk_test, SE_bulk_train, SE_bulk_test]
        label_list = ['JA_train','JA_test','DI_train','DI_test', 'SE_train','SE_test']       
        draw_curves(data_list = data_list, label_list = label_list, color_list = config.COLOR[0:6], filename = filename)
            
if __name__ == '__main__':
    main()

