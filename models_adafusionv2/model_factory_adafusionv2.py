# Author: Jacek Komorowski
# Warsaw University of Technology

# from models.minkloc import MinkLoc
# from models.minkloc_multimodal import ResNetFPNv2


from models.minkloc_multimodal import ResNetFPNv2
from models_adafusionv2.minkloc_minkfpn import MinkLocAdaFusion
from models_adafusionv2.imagefe_adafusionv2 import ResNetFPNAdaFusionV2


import torch.nn as nn
import torch

from tools.utils import set_seed
set_seed(7)
# from tools.options import Options
# args = Options().parse()




class AdaFusionV2(nn.Module):
    def __init__(self, image_fe, cloud_fe):
        super(AdaFusionV2, self).__init__()


        self.cloud_fe = cloud_fe


        self.image_fe = image_fe





    def forward(self, batch):
        y = {}
        if self.image_fe is not None:
            image_embedding, imagefe_output_dict = self.image_fe(batch)
            assert image_embedding.dim() == 2
            y['image_embedding'] = image_embedding
            for _k, _v in imagefe_output_dict.items():
                y[_k] = _v

        if self.cloud_fe is not None:
            cloud_embedding, cloudfe_output_dict = self.cloud_fe(batch)
            assert cloud_embedding.dim() == 2
            y['cloud_embedding'] = cloud_embedding


        assert cloud_embedding.shape[0] == image_embedding.shape[0]

        image_atten = imagefe_output_dict['image_atten']
        cloud_atten = cloudfe_output_dict['cloud_atten']
        vl_atten = torch.cat([image_atten, cloud_atten], dim=1)

        y['embedding'] = vl_atten

        
        return y
















def model_factory_adafusionv2(
                cloud_fe_size, image_fe_size, 
                cloud_planes, cloud_layers, cloud_topdown,
                image_useallstages, image_fe, 
                useattenres
                ):



    cloud_fe = MinkLocAdaFusion(in_channels=1, feature_size=cloud_fe_size, output_dim=cloud_fe_size,
                        planes=cloud_planes, layers=cloud_layers, num_top_down=cloud_topdown,
                        conv0_kernel_size=5, block='ECABasicBlock', pooling_method='GeM',
                        useattenres=useattenres)
    




    # image_fe = ResnetFPN(out_channels=image_fe_size, lateral_dim=image_fe_size,
    #                     fh_num_bottom_up=4, fh_num_top_down=0,
    #                     add_basicblock=resnetfpn_add_basicblock)
    image_fe = ResNetFPNAdaFusionV2(
                        image_fe=image_fe,
                        image_pool_method='GeM',
                        image_useallstages=image_useallstages,
                        output_dim=image_fe_size,
                        useattenres=useattenres
    )


    model = AdaFusionV2(image_fe=image_fe, cloud_fe=cloud_fe)




    return model
