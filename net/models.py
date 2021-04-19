import torch
import torch.nn as nn
import torch.nn.functional as F
import net.xception as xception

from .ASPP import ASPP
from .convs import SeparableConv2d
from .modules import Lesion_Location_Mining
from .modules import Category_guided_Feature_Generation
from .modules import Global_Prototypes_Generator

class DSI_Net(nn.Module):
    def __init__(self, config, K=100):
        super(DSI_Net, self).__init__()
        self.backbone = None
        self.backbone_layers = None
        self.dropout = nn.Dropout(0.5)
        self.upsample_sub_x2 = nn.UpsamplingBilinear2d(scale_factor=2) 
        self.upsample_sub_x4 = nn.UpsamplingBilinear2d(scale_factor=4)         
        self.shortcut_conv = nn.Sequential(nn.Conv2d(256, 48, 1, 1, padding=1//2, bias=True),
				nn.BatchNorm2d(48),
				nn.ReLU(inplace=True),
		)
        self.aspp = ASPP(dim_in=2048, dim_out=256, rate=16//16, bn_mom = 0.99)
        self.coarse_head = nn.Sequential(
				nn.Conv2d(256+48, 256, 3, 1, padding=1, bias=True),
				nn.BatchNorm2d(256),
				nn.ReLU(inplace=True),
				nn.Dropout(0.5),
				nn.Conv2d(256, 256, 3, 1, padding=1, bias=True),
				nn.BatchNorm2d(256),
				nn.ReLU(inplace=True),
				nn.Dropout(0.1),
                nn.Conv2d(256, 2, kernel_size=1, stride=1, padding=0, bias=True)
		) 
        
        self.fine_head = nn.Sequential(
				nn.Conv2d(256+64+48, 256, 3, 1, padding=1, bias=True),
				nn.BatchNorm2d(256),
				nn.ReLU(inplace=True),
				nn.Dropout(0.5),
				nn.Conv2d(256, 256, 3, 1, padding=1, bias=True),
				nn.BatchNorm2d(256),
				nn.ReLU(inplace=True),
				nn.Dropout(0.1),
                nn.Conv2d(256, 2, kernel_size=1, stride=1, padding=0, bias=True)
		)

        self.cls_head = nn.Sequential(
			SeparableConv2d(1024, 1536, 3, dilation=2, stride=1, padding=2, bias=False),
			nn.BatchNorm2d(1536),
            nn.ReLU(inplace=True),
			SeparableConv2d(1536, 1536, 3, dilation=2, stride=1, padding=2, bias=False),
			nn.BatchNorm2d(1536),
			nn.ReLU(inplace=True),
			SeparableConv2d(1536, 2048, 3, dilation=2, stride=1, padding=2, bias=False),
			nn.BatchNorm2d(2048),
			nn.ReLU(inplace=True))     
        self.LLM = Lesion_Location_Mining(config, 1024, K)
        self.GPG = Global_Prototypes_Generator(2048, config.INTERMIDEATE_NUM)
        self.CFG = Category_guided_Feature_Generation(256, config.INTERMIDEATE_NUM, config.EM_STEP)        
        self.avgpool = nn.AdaptiveAvgPool2d(1)        
        self.cls_predict = nn.Linear(2048, config.NUM_CLASSES_CLS, bias = False)       
                       
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        self.backbone = xception.Xception(os = config.OS)
        self.backbone_layers = self.backbone.get_layers()


    def forward(self, x):
        x = self.backbone(x)
        
        #shllow feature
        layers = self.backbone.get_layers()
        feature_shallow = self.shortcut_conv(layers[0])    
        feature_aspp = self.aspp(layers[-1])
                
        #coarse seg        
        feature_coarse= self.dropout(feature_aspp)
        feature_coarse = self.upsample_sub_x2(feature_coarse)
        feature_coarse = torch.cat([feature_coarse,feature_shallow],1)
        seg_coarse = self.coarse_head(feature_coarse) 

        #####cls        
        cls_feats = layers[-2]
        b, c, h, w = cls_feats.size()
        mask_coarse = torch.softmax(seg_coarse, dim = 1)       
        mask_coarse = F.interpolate(mask_coarse, size=(h, w), mode="bilinear", align_corners=False)
        
        cls_feats = self.LLM(cls_feats, mask_coarse)               
        cls_feats = self.cls_head(cls_feats)                     
        cls_out = self.avgpool(cls_feats)         
        cls_out = cls_out.view(b, -1) 
        cls_out = self.cls_predict(cls_out)
               
        #fine seg
        global_prototypes = self.GPG(self.cls_predict.weight.detach(), cls_out.detach())
        context= self.CFG(feature_aspp, mask_coarse, global_prototypes)
        context = self.upsample_sub_x2(context)                 
        feature_fine= self.dropout(feature_aspp)
        feature_fine = self.upsample_sub_x2(feature_fine)
        feature_fine = torch.cat([feature_fine,context,feature_shallow],1)
        seg_fine = self.fine_head(feature_fine)          
        
        #final seg
        seg_coarse = self.upsample_sub_x4(seg_coarse)        
        seg_fine = self.upsample_sub_x4(seg_fine)        

        return seg_coarse, seg_fine, cls_out


        
        
        
        