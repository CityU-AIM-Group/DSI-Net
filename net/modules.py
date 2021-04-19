import torch
import torch.nn as nn

from .gumbel import Gumbel

class Category_guided_Feature_Generation(nn.Module):

    def __init__(self, 
                 in_channels = 256, 
                 out_channels = 64, EM_STEP = 3):
        super(Category_guided_Feature_Generation, self).__init__()
        self.out_channels = out_channels
        self.EM_STEP = EM_STEP
        self.conv0 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1),
                                nn.BatchNorm2d(out_channels),
                                nn.ReLU(True),
                                nn.Dropout2d(0.2, False),
                                nn.Conv2d(out_channels, out_channels, 1),
                                nn.BatchNorm2d(out_channels),
                                nn.ReLU(True),
                                nn.Dropout2d(0.1, False),                                
                                )
        
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1),
                                nn.BatchNorm2d(out_channels),
                                nn.ReLU(True),
                                nn.Dropout2d(0.2, False),
                                nn.Conv2d(out_channels, out_channels, 1),
                                nn.BatchNorm2d(out_channels),
                                nn.ReLU(True),
                                nn.Dropout2d(0.1, False),                                
                                )        
                
        self.conv2 = nn.Sequential(nn.Conv2d(out_channels*2, out_channels, 1),
                                nn.BatchNorm2d(out_channels),
                                nn.ReLU(True),
                                nn.Dropout2d(0.2, False)
                                )
      
    def forward(self, x, coarse_mask, global_prototypes, regular = 0.5):
        
        b, h, w = x.size(0), x.size(2), x.size(3)
        classes_num = coarse_mask.size(1)
        feats = self.conv0(x)     
        pseudo_mask = coarse_mask.view(b, classes_num, -1)
        feats = feats.view(b, self.out_channels, -1).permute(0, 2, 1) 
        # EM                    
        T = self.EM_STEP
        for t in range(T):                          
            prototypes = torch.bmm(pseudo_mask, feats)
            prototypes = prototypes / (1e-8 + prototypes.norm(dim=1, keepdim=True))            
            attention = torch.bmm(prototypes, feats.permute(0, 2, 1))
            attention = (self.out_channels**-regular) * attention
            pseudo_mask = torch.softmax(attention, dim=1) 
            pseudo_mask = pseudo_mask / (1e-8 + pseudo_mask.sum(dim=1, keepdim=True))            
        context_l = torch.bmm(prototypes.permute(0, 2, 1), pseudo_mask).view(b, self.out_channels, h, w)      

        feats = self.conv1(x)    
        feats = feats.view(b, self.out_channels, -1).permute(0, 2, 1)
        global_prototypes = global_prototypes / (1e-8 + global_prototypes.norm(dim=1, keepdim=True))
        
        #EM
        T = self.EM_STEP
        for t in range(T):                          
            attention = torch.bmm(global_prototypes, feats.permute(0, 2, 1))          
            attention = (self.out_channels**-regular) * attention
            pseudo_mask = torch.softmax(attention, dim=1)
            pseudo_mask = pseudo_mask / (1e-8 + pseudo_mask.sum(dim=1, keepdim=True))            
            global_prototypes = torch.bmm(pseudo_mask, feats)
            global_prototypes = global_prototypes / (1e-8 + global_prototypes.norm(dim=1, keepdim=True))            
        context_g = torch.bmm(global_prototypes.permute(0, 2, 1), pseudo_mask).view(b, self.out_channels, h, w) # b, 64, 56*56        
        
        context = torch.cat((context_l, context_g), dim = 1)
        context = self.conv2(context)
                   
        return context
    
    
class Global_Prototypes_Generator(nn.Module):

    def __init__(self, 
                 in_channels = 2048, 
                 out_channels = 64):
        super(Global_Prototypes_Generator, self).__init__()
        self.out_channels = out_channels
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1),
                                nn.BatchNorm2d(out_channels),
                                nn.ReLU(True),
                                nn.Conv2d(out_channels, out_channels, 1)                               
                                )
         
    def forward(self, prototypes, category):       
        classes_num, c = prototypes.size(0), prototypes.size(1)
        prototypes = prototypes.view(classes_num,c, 1, 1)
        prototypes = self.conv1(prototypes).view(classes_num,self.out_channels)             
        category = torch.softmax(category, dim = 1)
        b = category.size(0)
        bg_prototypes = prototypes[0]
        bg_prototypes = bg_prototypes.repeat(b, 1, 1)
        fg_prototypes = category[:,1:].view(b, classes_num-1, 1) * prototypes[1:]
        prototypes = torch.cat((bg_prototypes, fg_prototypes), dim = 1)
                       
        return prototypes    




class Binary_Gate_Unit(nn.Module):
    
    def __init__(self, config, in_channels = 1024,  k = 100):
        super(Binary_Gate_Unit, self).__init__()
        self.in_channels = in_channels
        self.k = k
        self.conv= nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=1, kernel_size=1, stride=1, padding=0, bias = False),
            nn.ReLU(inplace=True)            
            )
        self.fc1 = nn.Linear(k, int(torch.ceil(torch.tensor(k/2))))
        self.fc2 = nn.Linear(int(torch.ceil(torch.tensor(k/2))), k)
        
        self.gumbel = Gumbel(config)

    def forward(self,topk_prototypes):
        
        b = topk_prototypes.size(0)
        proto_weights = self.conv(topk_prototypes) # meta learner
        proto_weights = proto_weights.view(b, -1)
        proto_weights = self.fc1(torch.relu(proto_weights)) 
        proto_weights = self.fc2(proto_weights) # b, k
        proto_weights = self.gumbel(proto_weights)      
        proto_weights = proto_weights.view(b, 1, self.k, 1)
        
        return proto_weights


class Lesion_Location_Mining(nn.Module):

    def __init__(self, config, in_channels = 1024,  k = 100):
        super(Lesion_Location_Mining, self).__init__()
        self.k = k       
        self.BGU_fore =Binary_Gate_Unit(config, in_channels = in_channels,  k = k)
        self.BGU_back =Binary_Gate_Unit(config, in_channels = in_channels,  k = k)
        
    def forward(self, feats, soft_mask):
        
        b,c,h,w = feats.size()
        hard_mask = torch.max(soft_mask, dim = 1, keepdim = True)[1] # b, 1, h, w
        background_hard_mask = (hard_mask == 0).float()          
        foreground_hard_mask = (hard_mask == 1).float()
        assert torch.sum(hard_mask == 2) == 0, 'Error in Lesion_Location_Mining_Module'
        background_soft_mask, foreground_soft_mask = soft_mask.split(1, dim = 1)    #b, 1, h, w
        foreground_feats = feats * foreground_hard_mask # b, c, h, w
        background_feats = feats * background_hard_mask # b, c, h, w    
        feats = feats.view(b, c, -1) # b, c, hw        
        
        #****** foreground-->background **********#
        #key generator
        foreground_soft_mask = foreground_soft_mask.view(b, 1, -1)
        topk_idx = torch.topk(foreground_soft_mask, self.k, dim = -1, largest=True)[1]     
        topk_prototypes = []
        for i in range(b):
            feats_temp = feats[i,:,topk_idx[i]] # c, k
            topk_prototypes.append(feats_temp)
        topk_prototypes = torch.stack(topk_prototypes) # b, c, k
        topk_prototypes = topk_prototypes.view(b, c, self.k, 1)                
        proto_weights = self.BGU_fore(topk_prototypes) 
        topk_prototypes = topk_prototypes * proto_weights # b, c, k, 1         
               
        # b, c, h, w # b, c, k ---> 
        background_feats = background_feats.view(b, c, -1) # b, c, hw
        topk_prototypes = topk_prototypes.view(b, c, -1).permute(0, 2, 1) # b, k, c 
        fore_attention_map = torch.matmul(topk_prototypes, background_feats) # b, k ,hw
       
        #norm + relu
        norm_prototypes = torch.norm(topk_prototypes, dim = -1, keepdim=True) # b, k, 1
        norm_background_feats = torch.norm(background_feats, dim = 1, keepdim=True) #b, 1, hw
        norm = torch.bmm(norm_prototypes, norm_background_feats) # b, k, hw
        fore_attention_map =  fore_attention_map /(norm + 1e-8)
        fore_attention_map = torch.relu(fore_attention_map)
        fore_attention_map = fore_attention_map.view(b, self.k, h, w) 
        fore_attention_map = torch.max(fore_attention_map, dim = 1,  keepdim = True) [0]

        #****** background-->foreground**********#
        #key generator
        background_soft_mask = background_soft_mask.view(b, 1, -1)
        topk_idx = torch.topk(background_soft_mask, self.k, dim = -1, largest=True)[1]                 
        topk_prototypes = []
        for i in range(b):
            feats_temp = feats[i,:,topk_idx[i]] # c, k
            topk_prototypes.append(feats_temp)
        topk_prototypes = torch.stack(topk_prototypes) # b, c, k
        topk_prototypes = topk_prototypes.view(b, c, self.k, 1)        
        proto_weights = self.BGU_back(topk_prototypes) 
        topk_prototypes = topk_prototypes * proto_weights # b, c, k, 1
               
        # b, c, h, w # b, c, k ---> 
        foreground_feats = foreground_feats.view(b, c, -1) # b, c, hw
        topk_prototypes = topk_prototypes.view(b, c, -1).permute(0, 2, 1) # b, k, c
        back_attention_map = torch.matmul(topk_prototypes, foreground_feats) # b, k ,hw
        #norm + relu
        norm_prototypes = torch.norm(topk_prototypes, dim = -1, keepdim=True) # b, k, 1
        norm_foreground_feats = torch.norm(foreground_feats, dim = 1, keepdim=True) #b, 1, hw
        norm = torch.bmm(norm_prototypes, norm_foreground_feats) # b, k, hw
        back_attention_map =  back_attention_map /(norm + 1e-8)
        back_attention_map = torch.relu(back_attention_map)
        back_attention_map = back_attention_map.view(b, self.k, h, w) 
        back_attention_map = torch.max(back_attention_map, dim = 1,  keepdim = True) [0]
     
        #merging
        feats = feats.view(b, c, h, w)
        foreground_soft_mask = foreground_soft_mask.view(b, 1, h, w)
        feats = feats + feats * (foreground_soft_mask - back_attention_map + fore_attention_map)
        
        return feats  #b, c, h,w    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    