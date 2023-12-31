




import torch
import torch.nn as nn
import torchvision.models as TVmodels
from network.image_pool_fns import ImageGeM
from network.image_pool_fns import ImageCosPlace
from network.image_pool_fns import ImageNetVLAD
from network.image_pool_fns import ImageConvAP

import torch.nn.functional as F

from tools.utils import set_seed
set_seed(7)




# ----------------------------------  ResNetFPNv2 ----------------------------------
class ResNetFPNAdaFusionV2(torch.nn.Module):
    def __init__(self, 
                 image_fe,
                 image_pool_method, # GeM
                 image_useallstages, # True
                 output_dim,
                 useattenres
                 ):
        super().__init__()
        '''
        resnet [64,64,128,256,512]
        convnext [96,96,192,384,768]
        swin [96,96,192,384,768]
        swin_v2 [96,96,192,384,768]
        '''


        self.image_fe = image_fe
        self.image_pool_method = image_pool_method




        self.image_useallstages = image_useallstages
        self.useattenres = useattenres

        # -- resnet
        if self.image_fe == 'resnet18':
            self.model = TVmodels.resnet18(weights='IMAGENET1K_V1')
            if self.image_useallstages:
                self.last_dim = 512
            else:
                self.last_dim = 256
        elif self.image_fe == 'resnet34':
            self.model = TVmodels.resnet34(weights='IMAGENET1K_V1')
            if self.image_useallstages:
                self.last_dim = 512
            else:
                self.last_dim = 256
        elif self.image_fe == 'resnet50':
            self.model = TVmodels.resnet50(weights='IMAGENET1K_V2')
            if self.image_useallstages:
                self.last_dim = 2048
            else:
                self.last_dim = 1024
        elif self.image_fe == 'resnet101':
            self.model = TVmodels.resnet101(weights='IMAGENET1K_V2')
            if self.image_useallstages:
                self.last_dim = 2048
            else:
                self.last_dim = 1024
        elif self.image_fe == 'resnet152':
            self.model = TVmodels.resnet152(weights='IMAGENET1K_V2')
            if self.image_useallstages:
                self.last_dim = 2048
            else:
                self.last_dim = 1024


        # -- convnext
        elif self.image_fe == 'convnext_tiny':
            self.model = TVmodels.convnext_tiny(weights='IMAGENET1K_V1')
            if self.image_useallstages:
                self.last_dim = 768
            else:
                self.last_dim = 384
        elif self.image_fe == 'convnext_small':
            self.model = TVmodels.convnext_small(weights='IMAGENET1K_V1')
            if self.image_useallstages:
                self.last_dim = 768
            else:
                self.last_dim = 384


        # -- swin
        elif self.image_fe == 'swin_t':
            self.model = TVmodels.swin_t(weights='IMAGENET1K_V1')
            if self.image_useallstages:
                self.last_dim = 768
            else:
                self.last_dim = 384
        elif self.image_fe == 'swin_s':
            self.model = TVmodels.swin_s(weights='IMAGENET1K_V1')
            self.last_dim = 384
        elif self.image_fe == 'swin_v2_t':
            self.model = TVmodels.swin_v2_t(weights='IMAGENET1K_V1')
            if self.image_useallstages:
                self.last_dim = 768
            else:
                self.last_dim = 384
        elif self.image_fe == 'swin_v2_s':
            self.model = TVmodels.swin_v2_s(weights='IMAGENET1K_V1')
            self.last_dim = 384





        self.conv1x1 = nn.Conv2d(self.last_dim, output_dim, kernel_size=1)


        self.image_gem = ImageGeM() # *1
        self.imagecosplace =  ImageCosPlace(output_dim, output_dim) # *1
        self.imageconvap = ImageConvAP(output_dim, output_dim) # *4
        self.imagenetvlad = ImageNetVLAD(clusters_num=64,
                                         dim=output_dim) # *4



        attn_input_dim = self.last_dim + self.last_dim//2 + self.last_dim//4
        self.attn = nn.Sequential(
            nn.Linear(attn_input_dim, output_dim),
            nn.ReLU(inplace=True),
            nn.Linear(output_dim, 1),

            # nn.Linear(attn_input_dim, 1),
            # nn.Sigmoid()
        )
        

    def forward_resnet(self, x):
        fe_output_dict = {}
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x_avgpool = F.avg_pool2d(x, kernel_size=x.size()[2:]).squeeze(3).squeeze(2)
        fe_output_dict['image_layer1'] = x
        fe_output_dict['image_layer1_avgpool'] = x_avgpool

        x = self.model.layer2(x)
        x_avgpool = F.avg_pool2d(x, kernel_size=x.size()[2:]).squeeze(3).squeeze(2)
        fe_output_dict['image_layer2'] = x
        fe_output_dict['image_layer2_avgpool'] = x_avgpool

        x = self.model.layer3(x)
        x_avgpool = F.avg_pool2d(x, kernel_size=x.size()[2:]).squeeze(3).squeeze(2)
        fe_output_dict['image_layer3'] = x
        fe_output_dict['image_layer3_avgpool'] = x_avgpool

        if self.image_useallstages:
            x = self.model.layer4(x)
            x_avgpool = F.avg_pool2d(x, kernel_size=x.size()[2:]).squeeze(3).squeeze(2)
            fe_output_dict['image_layer4'] = x
            fe_output_dict['image_layer4_avgpool'] = x_avgpool

        return x, fe_output_dict



    def forward_convnext(self, x):
        layers_list = list(self.model.features.children())
        assert len(layers_list)==8
        if not self.image_useallstages:
            layers_list = layers_list[:-2]
        else:
            layers_list = layers_list

        for i in range(len(layers_list)):
            layer = layers_list[i]
            x = layer(x)
        return x

    
    def forward_swin(self, x):
        layers_list = list(self.model.features.children())
        if not self.image_useallstages:
            layers_list = layers_list[:-2]
        else:
            layers_list = layers_list
        for i in range(len(layers_list)):
            layer = layers_list[i]
            x = layer(x)
        x = x.permute(0,3,1,2)
        return x






    def forward(self, data_dict):


        x = data_dict['images']
        fe_output_dict = {}

        
        if self.image_fe in ['resnet18','resnet34','resnet50','resnet101','resnet152']:
            x, fe_output_dict = self.forward_resnet(x)
        elif self.image_fe in ['convnext_tiny','convnext_small']:
            x = self.forward_convnext(x)
        elif self.image_fe in ['swin_t','swin_s']:
            x = self.forward_swin(x)
        elif self.image_fe in ['swin_v2_t','swin_v2_s']:
            x = self.forward_swin(x)
        else:
            raise NotImplementedError
        

        x_feat_256 = x
        x_feat_256 = self.conv1x1(x_feat_256)


        if self.image_pool_method == 'GeM':
            embedding = self.image_gem(x_feat_256)

        elif self.image_pool_method == 'ConvAP':
            embedding = self.imageconvap(x_feat_256)

        elif self.image_pool_method == 'CosPlace':
            embedding = self.imagecosplace(x_feat_256)

        elif self.image_pool_method == 'NetVLAD':
            embedding = self.imagenetvlad(x_feat_256)
        
        else:
            raise NotImplementedError
        


        image_layer1_avgpool = fe_output_dict['image_layer1_avgpool']
        image_layer2_avgpool = fe_output_dict['image_layer2_avgpool']
        image_layer3_avgpool = fe_output_dict['image_layer3_avgpool']

        image_atten = torch.cat([
            image_layer1_avgpool, 
            image_layer2_avgpool, 
            image_layer3_avgpool
            ], dim=1)
        
        image_atten = self.attn(image_atten) # [b,1]
        image_atten = embedding * image_atten # [b,outputdim]
        if self.useattenres:
            image_atten += embedding 

        fe_output_dict['image_atten'] = image_atten

        return embedding, fe_output_dict
    

    




