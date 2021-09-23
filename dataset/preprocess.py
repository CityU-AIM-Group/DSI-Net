# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 16:36:36 2020

@author: meiluzhu
"""

import os
import numpy as np
import pickle
import cv2

m_type = ['vascularlesions','inflammatory','normal']
patients = []
images = []
labels = []
res = 288

#### CAD-CAP
base = 'C:\\ZML\\Dataset\\WCE\\CAD-CAP'
save_base = 'C:\\ZML\\Dataset\\WCE\\temp\\CAD-CAP'

file = os.listdir(os.path.join(base, m_type[1]))
for f in file:
    filename = f.split('.')
    if os.path.exists(os.path.join(base,m_type[1],filename[0])+'_a'+'.jpg'):                
        img_dir = m_type[1]+'\\'+ f
        mask_dir = m_type[1]+'\\'+ filename[0]+'_a.jpg'
        print(img_dir)
        image = cv2.imread(os.path.join(base, img_dir))
        mask = cv2.imread(os.path.join(base, mask_dir),cv2.IMREAD_GRAYSCALE)
        h,w,c = image.shape
        h_top = 0
        h_bot = h
        w_lef = 0
        w_rig = w              
        if h == 576 or h == 704: 
            h_top = 32
            h_bot = h-32
            w_lef = 32
            w_rig = w-32
        post_img = image[h_top:h_bot,w_lef:w_rig,:]
        post_img[0:45,0:15,:] = 0
        post_mask = mask[h_top:h_bot,w_lef:w_rig]
        post_img = cv2.resize(post_img, (res,res), interpolation=cv2.INTER_NEAREST)
        post_mask = cv2.resize(post_mask, (res,res), interpolation=cv2.INTER_NEAREST) 
        new_dir = os.path.join(save_base,img_dir)
        cv2.imwrite(new_dir, post_img)
        new_dir = os.path.join(save_base,mask_dir)
        cv2.imwrite(new_dir, post_mask)        
        post_mask[post_mask>0] = 1                           
        patient = {'image':'CAD-CAP/'+img_dir.replace('\\','/' ), 'mask': 'CAD-CAP/'+mask_dir.replace('\\','/' ),'label': 2} 
        patients.append(patient)
          
file = os.listdir(os.path.join(base, m_type[0]))
for f in file:
    filename = f.split('.')
    if os.path.exists(os.path.join(base,m_type[0],filename[0])+'_a'+'.jpg'):
        img_dir = m_type[0]+'/'+ f
        mask_dir = m_type[0]+'/'+ filename[0]+'_a.jpg'
        print(img_dir)       
        image = cv2.imread(os.path.join(base, img_dir))
        mask = cv2.imread(os.path.join(base, mask_dir),cv2.IMREAD_GRAYSCALE)
        ######
        h,w,c = image.shape
        h_top = 0
        h_bot = h
        w_lef = 0
        w_rig = w              
        if h == 576 or h == 704: 
            h_top = 32
            h_bot = h-32
            w_lef = 32
            w_rig = w-32
        post_img = image[h_top:h_bot,w_lef:w_rig,:]
        h,w,c = post_img.shape
        if h>600:
            post_img[0:10,0:139,:] = 0
            post_img[h-2:h,0:191,:] = 0
            post_img[h-5:h,0:150,:] = 0
        else:
            post_img[0:10,0:115,:] = 0 
            post_img[h-2:h,0:133,:] = 0
            post_img[h-5:h,0:110,:] = 0                        
        post_img[0:60,0:21,:] = 0
        post_mask = mask[h_top:h_bot,w_lef:w_rig]
        post_img = cv2.resize(post_img, (res,res), interpolation=cv2.INTER_NEAREST)
        post_mask = cv2.resize(post_mask, (res,res), interpolation=cv2.INTER_NEAREST) 
        new_dir = os.path.join(save_base,img_dir)
        cv2.imwrite(new_dir, post_img)
        new_dir = os.path.join(save_base,mask_dir)
        cv2.imwrite(new_dir, post_mask)          
        post_mask[post_mask>0] = 1 
        patient = {'image':'CAD-CAP/'+img_dir.replace('\\','/' ), 'mask': 'CAD-CAP/'+mask_dir.replace('\\','/' ),'label': 1} 
        patients.append(patient) 

file = os.listdir(os.path.join(base, m_type[2]))
for f in file:
    img_dir = m_type[2]+'/'+ f
    print(img_dir)       
    image = cv2.imread(os.path.join(base, img_dir))
    h,w,c = image.shape
    h_top = 0
    h_bot = h
    w_lef = 0
    w_rig = w              
    if h == 576 or h == 704: 
        h_top = 32
        h_bot = h-32
        w_lef = 32
        w_rig = w-32
    post_img = image[h_top:h_bot,w_lef:w_rig,:]
    h,w,c = post_img.shape
    post_img[0:45,0:15,:] = 0                         
    post_img = cv2.resize(post_img, (res,res), interpolation=cv2.INTER_NEAREST)
    img_dir = img_dir.replace(' ', '_')
    new_dir = os.path.join(save_base,img_dir)
    cv2.imwrite(new_dir, post_img)
    post_mask = np.zeros((res,res), dtype = np.uint8)
    filename = img_dir.split('.')
    mask_dir = filename[0]+'_a.jpg'
    new_dir = os.path.join(save_base,mask_dir)
    cv2.imwrite(new_dir, post_mask)          
    post_mask[post_mask>0] = 1 
    patient = {'image':'CAD-CAP/'+img_dir.replace('\\','/' ), 'mask': 'CAD-CAP/'+mask_dir.replace('\\','/' ),'label': 0} 
    patients.append(patient) 

####KID
base = 'C:\\ZML\\Dataset\\WCE\\KID'
save_base = 'C:\\ZML\\Dataset\\WCE\\temp\\KID'
era = 4
file = os.listdir(os.path.join(base, m_type[1]))
for f in file:
    filename = f.split('.')
    if os.path.exists(os.path.join(base,m_type[1],filename[0])+'m'+'.png'):                
        img_dir = m_type[1]+'\\'+ f
        mask_dir = m_type[1]+'\\'+ filename[0]+'m.png'
        print(img_dir)
        image = cv2.imread(os.path.join(base, img_dir))
        mask = cv2.imread(os.path.join(base, mask_dir),cv2.IMREAD_GRAYSCALE)
        h,w,c = image.shape
        h_top = 20 + era
        h_bot = h - 20 - era
        w_lef = 20 + era
        w_rig = w - 20 - era        
        post_img = image[h_top:h_bot,w_lef:w_rig,:]
        post_mask = mask[h_top:h_bot,w_lef:w_rig]        
        post_img = cv2.resize(post_img, (res,res), interpolation=cv2.INTER_NEAREST)
        post_mask = cv2.resize(post_mask, (res,res), interpolation=cv2.INTER_NEAREST) 
        new_dir = os.path.join(save_base,img_dir)
        cv2.imwrite(new_dir, post_img)
        new_dir = os.path.join(save_base,mask_dir)
        cv2.imwrite(new_dir, post_mask)        
        post_mask[post_mask>0] = 1                 
        patient = {'image':'KID/'+img_dir.replace('\\','/' ), 'mask': 'KID/'+mask_dir.replace('\\','/' ),'label': 2} 
        patients.append(patient)
            
file = os.listdir(os.path.join(base, m_type[0]))
for f in file:
    filename = f.split('.')
    if os.path.exists(os.path.join(base,m_type[0],filename[0])+'m'+'.png'):
        img_dir = m_type[0]+'/'+ f
        mask_dir = m_type[0]+'/'+ filename[0]+'m.png'
        print(img_dir)       
        image = cv2.imread(os.path.join(base, img_dir))
        mask = cv2.imread(os.path.join(base, mask_dir),cv2.IMREAD_GRAYSCALE)
        h,w,c = image.shape
        h_top = 20 + era
        h_bot = h - 20 - era
        w_lef = 20 + era
        w_rig = w - 20 - era               
        post_img = image[h_top:h_bot,w_lef:w_rig,:]                         
        post_mask = mask[h_top:h_bot,w_lef:w_rig]
        post_img = cv2.resize(post_img, (res,res), interpolation=cv2.INTER_NEAREST)
        post_mask = cv2.resize(post_mask, (res,res), interpolation=cv2.INTER_NEAREST) 
        new_dir = os.path.join(save_base,img_dir)
        cv2.imwrite(new_dir, post_img)
        new_dir = os.path.join(save_base,mask_dir)
        cv2.imwrite(new_dir, post_mask)            
        post_mask[post_mask>0] = 1 
        patient = {'image':'KID/'+img_dir.replace('\\','/' ), 'mask': 'KID/'+mask_dir.replace('\\','/' ),'label': 1} 
        patients.append(patient) 

m_type[2] = 'normal-small-bowel'
file = os.listdir(os.path.join(base, m_type[2]))
for f in file:
    img_dir = m_type[2]+'/'+ f
    print(img_dir)       
    image = cv2.imread(os.path.join(base, img_dir))
    h,w,c = image.shape
    h_top = 20 + era
    h_bot = h - 20 - era
    w_lef = 20 + era
    w_rig = w - 20 - era
    post_img = image[h_top:h_bot,w_lef:w_rig,:]          
    post_img = cv2.resize(post_img, (res,res), interpolation=cv2.INTER_NEAREST)
    img_dir = img_dir.replace(' ', '_')
    img_dir = img_dir.replace('normal-small-bowel', 'normal')    
    new_dir = os.path.join(save_base,img_dir)
    cv2.imwrite(new_dir, post_img)
    post_mask = np.zeros((res,res), dtype = np.uint8)
    filename = img_dir.split('.')
    mask_dir = filename[0]+'m.png'
    new_dir = os.path.join(save_base,mask_dir)
    cv2.imwrite(new_dir, post_mask)          
    post_mask[post_mask>0] = 1 
    patient = {'image':'KID/'+img_dir.replace('\\','/' ), 'mask': 'KID/'+mask_dir.replace('\\','/' ),'label': 0} 
    patients.append(patient) 

np.random.shuffle(patients)
trainset = patients[0:2470]
testset = patients[2470:]
dataset = {'train': trainset, 'test': testset}
path = os.path.join('C:\\ZML\\Dataset\\WCE\\temp', 'WCE_Dataset_larger_Fold1.pkl')
if os.path.exists(path):
    os.remove(path)
with open(path,'wb') as f:
    pickle.dump(dataset, f)                      

        
        