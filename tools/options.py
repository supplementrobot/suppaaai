import argparse
import os



# rkdg 0.1    1    10
# ours 0.1    1


class Options():
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    def initialize(self):



        self.parser.add_argument('--cuda', type=str, default='0')
        self.parser.add_argument('--tryid', type=int, default=0)
        self.parser.add_argument('--save_weights', type=str, default=False)
        self.parser.add_argument('--save_epoch', type=int, default=-1)
        self.parser.add_argument('--num_workers', type=int, default=8)
        # oxford   boreas
        self.parser.add_argument('--dataset', type=str, 
                                 default='boreas'
                                 )
        self.parser.add_argument('--dataset_folder', type=str, 
                                 default='/home/BenchmarkBoreasv3', # replace with your own path
                                 )
        self.parser.add_argument('--image_path', type=str, 
                                 default='/home/BenchmarkBoreasv3/boreas', # replace with your own path
                                 )
        self.parser.add_argument('--n_points_boreas', type=int, default=4096) # only for boreas
        self.parser.add_argument('--oxford_quantization_size', type=float, default=0.01)
        self.parser.add_argument('--boreas_quantization_size', type=float, default=1)


        





        # ------------------------ multilayer loss weight
        self.parser.add_argument('--distil_logit_weight', type=float, default=1)
        self.parser.add_argument('--distil_layer1_weight', type=float, default=0)
        self.parser.add_argument('--distil_layer2_weight', type=float, default=0)
        self.parser.add_argument('--distil_layer3_weight', type=float, default=0)

        self.parser.add_argument('--task_logit_weight', type=float, default=1)
        self.parser.add_argument('--task_layer1_weight', type=float, default=0)
        self.parser.add_argument('--task_layer2_weight', type=float, default=0)
        self.parser.add_argument('--task_layer3_weight', type=float, default=0)


        self.parser.add_argument('--cdistil_layer1_weight', type=float, default=0)
        self.parser.add_argument('--cdistil_layer2_weight', type=float, default=0)
        self.parser.add_argument('--cdistil_layer3_weight', type=float, default=0)
        self.parser.add_argument('--cdistil_logitsimloss_weight', type=float, default=0)




        # ------------------------ selfagent
        self.parser.add_argument('--rkdgloss_weight', type=float, default=0.1) # integrated


        # ------------------------ crossagent
        # MSE   L1   
        self.parser.add_argument('--crosslogitdistloss_type', type=str, default='MSE')
        self.parser.add_argument('--crosslogitdistloss_logit_norm_fn', type=str, default='l2')
        self.parser.add_argument('--crosslogitdistloss_weight_st2ss', type=str, default='0.1') 
        self.parser.add_argument('--crosslogitdistloss_weight_tt2st', type=str, default='0')

        self.parser.add_argument('--crosslogitsimloss_type', type=str, default='MSE')
        self.parser.add_argument('--crosslogitsimloss_logit_norm_fn', type=str, default='l2')
        self.parser.add_argument('--crosslogitsimloss_weight_st2ss', type=str, default='0.1') 
        self.parser.add_argument('--crosslogitsimloss_weight_tt2st', type=str, default='0')
        self.parser.add_argument('--crosslogitsimloss_divide', type=str, default=True) 

        self.parser.add_argument('--crosslogitgeodistloss_type', type=str, default='MSE')
        self.parser.add_argument('--crosslogitgeodistloss_logit_norm_fn', type=str, default='l2')
        self.parser.add_argument('--crosslogitgeodistloss_weight_st2ss', type=str, default='0.1') 
        self.parser.add_argument('--crosslogitgeodistloss_weight_tt2st', type=str, default='0')
        self.parser.add_argument('--curvature', type=float, default=0.1) 


        # ------------------------ logit mapping
        # identity   topoincare   id_topoin
        self.parser.add_argument('--logit_mapping_fn', type=str, default='identity')




        # ------------------------ distil method
        self.parser.add_argument('--distil_method', type=str, default='ours_rkdg')
        # train_teacher   train_student   
        self.parser.add_argument('--train_mode', type=str, default='train_student')




        # ------------------------ student model
        # minkloc3dv2  prfusion.image_fe  resnet18   resnet34   minkloconly   
        self.parser.add_argument('--student_model', type=str, default='resnet18') # image fes allstage False
        self.parser.add_argument('--student_output_dim', type=int, default=384) # for resnet18, resnet34, minkloconly
        self.parser.add_argument('--student_minkloconly_planes', type=str, default='32_64_128') # 32_64_128+1_1_1+0  32_64_64+1_1_1+1
        self.parser.add_argument('--student_minkloconly_layers', type=str, default='1_1_1')
        self.parser.add_argument('--student_minkloconly_topdown', type=int, default=0)



        self.parser.add_argument('--teacher_weights_path', type=str, 
            # default='teacher_weights/boreas__T:resnet50__256__b128__trainteacher/r1.pth',
            # default='teacher_weights/boreas__T:resnet50__384__b128__trainteacher/r1.pth',
            # default='teacher_weights/boreas__T:minkloconly__32_64_128__1_1_1__0__256__b128__trainteacher/r1.pth',
            default='teacher_weights/boreas__T:minkloconly__32_64_64__1_1_1__1__384__b128__trainteacher/r1.pth',

        )





        # ------------------------ teacher model
        # prfusion   convnext_tiny   resnet18   resnet34   minkloc  minklocmmadd  adafusion  minkloc3dv2     
        # minklocmmcat   adafusionv2
        # minkloconly    resnet50
        self.parser.add_argument('--model', type=str, default='minkloconly')
        # ------------------------ prfusion
        # image_fe
        self.parser.add_argument('--image_fe', type=str, default='resnet18')
        self.parser.add_argument('--num_other_stage_blocks', type=int, default=3)
        self.parser.add_argument('--num_stage3_blocks', type=int, default=3)
        # cloud_fe
        self.parser.add_argument('--cloud_fe', type=str, default='minkloc')
        # ffbs
        self.parser.add_argument('--num_ffbs', type=int, default=2)
        self.parser.add_argument('--ffb_fusion_block_type', type=str, default='qkv')
        self.parser.add_argument('--num_blocks_in_ffb', type=int, default=1)
        self.parser.add_argument('--num_cloud_points', type=int, default=128)
        # later_branch
        self.parser.add_argument('--num_blocks_in_later_branch', type=int, default=1)
        self.parser.add_argument('--later_branch_block_type', type=str, default='basicblock')
        self.parser.add_argument('--later_branch_interaction_type', type=str, default='add')
        # output, sharing params
        self.parser.add_argument('--final_embedding_type', type=str, default='imagelater_cloudlater')
        self.parser.add_argument('--num_heads', type=int, default=8)

        # ------------------------ minklocmmadd / minklocmmcat
        self.parser.add_argument('--minklocmm_image_fe', type=str, default='resnet18') # resnet18
        self.parser.add_argument('--minklocmm_image_fe_dim', type=int, default=256) # 128
        self.parser.add_argument('--minklocmm_cloud_fe_dim', type=int, default=128) # 256
        self.parser.add_argument('--minklocmm_cloud_planes', type=str, default='32_64_64') # 32_64_64
        self.parser.add_argument('--minklocmm_cloud_layers', type=str, default='1_1_1')   # 1_1_1
        self.parser.add_argument('--minklocmm_cloud_topdown', type=int, default=1) # 1
        self.parser.add_argument('--minklocmm_image_useallstages', type=str, default=False)
        # resnet18   resnet34   resnet50  resnet101  resnet152  convnext_tiny   swin_t   swin_v2_t

        # ------------------------ adaptor 
        # 256   384   None
        self.parser.add_argument('--teacher_adaptor_dim', type=int, default=None)

        # ------------------------ adafusionv2
        self.parser.add_argument('--teacher_adav2_image_fe', type=str, default='resnet18') # resnet18
        self.parser.add_argument('--teacher_adav2_image_fe_dim', type=int, default=128) # 128 (128+128 not work)
        self.parser.add_argument('--teacher_adav2_cloud_fe_dim', type=int, default=128) # 256
        self.parser.add_argument('--teacher_adav2_cloud_planes', type=str, default='32_64_128') # 32_64_64
        self.parser.add_argument('--teacher_adav2_cloud_layers', type=str, default='1_1_1')   # 1_1_1
        self.parser.add_argument('--teacher_adav2_cloud_topdown', type=int, default=0) # 1
        self.parser.add_argument('--teacher_adav2_image_useallstages', type=str, default=False)
        self.parser.add_argument('--teacher_adav2_useattenres', type=str, default=False) # 

        # ------------------------ resnet/convnext
        self.parser.add_argument('--teacher_image_fe_dim', type=int, default=256) # resnet18

        # ------------------------ minkloconly
        self.parser.add_argument('--teacher_minkloconly_planes', type=str, default='32_64_64') # 32_64_64
        self.parser.add_argument('--teacher_minkloconly_layers', type=str, default='1_1_1') # 1_1_1
        self.parser.add_argument('--teacher_minkloconly_topdown', type=int, default=1) # 1
        self.parser.add_argument('--teacher_minkloconly_dim', type=int, default=384)







        # ---------------------------------------- train
        # ---------------------------------------- train

        # single   multi_sep   multi_joint   multi_step
        self.parser.add_argument('--train_step_type', type=str, default='single')
        self.parser.add_argument('--epochs', type=int, default=60)
        self.parser.add_argument('--train_batch_size', type=int, default=128)
        self.parser.add_argument('--train_batch_split_size', type=int, default=128)
        self.parser.add_argument('--val_batch_size', type=int, default=160)
        self.parser.add_argument('--image_lidar_k', type=int, default=1) # 1   None
        # Adam  SGD  CosineAnnealingLR  MultiStepLR  
        self.parser.add_argument('--optimizer', type=str, default='Adam')
        self.parser.add_argument('--image_lr', type=float, default=1e-4)
        self.parser.add_argument('--cloud_lr', type=float, default=1e-3)
        self.parser.add_argument('--weight_decay', type=float, default=1e-4)
        self.parser.add_argument('--milestones', type=str, default='40') # 40, 250_350
        # MultiStepLR
        self.parser.add_argument('--scheduler', type=str, default='MultiStepLR') # 40, 0.1
        # BatchHardTripletMarginLoss   MultiBatchHardTripletMarginLoss   TruncatedSmoothAP    MultiTruncatedSmoothAP
        self.parser.add_argument('--loss', type=str, default='MultiBatchHardTripletMarginLoss')
        self.parser.add_argument('--ap_tau1', type=float, default=0.01)
        self.parser.add_argument('--ap_similarity', type=str, default='euclidean') # euclidean  cosine
        self.parser.add_argument('--ap_positives_per_query', type=int, default=4)
        self.parser.add_argument('--weights', type=str, default='0.5_0.5_0') # 0.5 0.5 0
        self.parser.add_argument('--normalize_embeddings', type=str, default=False)
        self.parser.add_argument('--margin', type=float, default=0.2)




        # -- image augmentation rate
        self.parser.add_argument('--bcs_aug_rate', type=float, default=0.2) # 0.2
        self.parser.add_argument('--hue_aug_rate', type=float, default=0.1) # 0.1



        self.parser.add_argument('--config', type=str, default='config/config_baseline_multimodal.txt')
        self.parser.add_argument('--model_config', type=str, default='models/minklocmultimodal.txt')





        self.parser.add_argument('--resume_epoch', type=int, default=-1)




        self.parser.add_argument('--logdir', type=str, default='./logs')
        self.parser.add_argument('--exp_name', type=str, default='name')



        self.parser.add_argument('--models_dir', type=str, default='models')

        





    def parse(self):
        self.initialize()
        self.args = self.parser.parse_args()

        args_dict = vars(self.args)
        # print(args_dict)
        for k, v in args_dict.items():
            if v=='False':
                args_dict[k] = False
            elif v=='True':
                args_dict[k] = True
            elif v=='None':
                args_dict[k] = None
            
            if k=='weights':
                assert isinstance(v, str)
                weights = v.split('_')
                weights = [float(w) for w in weights]
                args_dict[k] = weights

        self.args = argparse.Namespace(**args_dict)






        # dataset params
        if self.args.dataset == 'oxford':    

            self.args.lidar2image_ndx_path = os.path.join(self.args.image_path, 'lidar2image_ndx.pickle')
            self.args.eval_database_files = ['oxford_evaluation_database.pickle']
            self.args.eval_query_files = ['oxford_evaluation_query.pickle']


        elif self.args.dataset == 'oxfordadafusion':

            self.args.lidar2image_ndx_path = os.path.join(self.args.image_path, 'oxfordadafusion_lidar2image_ndx.pickle')
            self.args.eval_database_files = ['oxfordadafusion_evaluation_database.pickle']
            self.args.eval_query_files = ['oxfordadafusion_evaluation_query.pickle']



        elif self.args.dataset == 'boreas':
            self.args.lidar2image_ndx_path = os.path.join(self.args.dataset_folder, 'boreas_lidar2image_ndx.pickle')
            self.args.eval_database_files = ['boreas_evaluation_database.pickle']
            self.args.eval_query_files = ['boreas_evaluation_query.pickle']

        else:
            raise Exception
        
        
        if self.args.dataset in ['oxford', 'oxfordadafusion']:
            self.args.num_points = 4096
        elif self.args.dataset == 'boreas':
            self.args.num_points = self.args.n_points_boreas
        else:
            raise Exception




        # -------------------- teacher --------------------
        self.args.exp_name = ''
        self.args.exp_name += f'{self.args.dataset}'



        # ---- teacher model
        self.args.exp_name += f'__T:{self.args.model}'

        if self.args.model == 'prfusion':
            # self.args.exp_name += f'++prf'
            self.args.exp_name += f'__{self.args.image_fe}'
            self.args.exp_name += f'__{self.args.final_embedding_type}'
        elif self.args.model in ['minklocmmadd', 'minklocmmcat']:
            # self.args.exp_name += f'++minkmm'
            self.args.exp_name += f'__{self.args.minklocmm_image_fe}'
            self.args.exp_name += f'__img{self.args.minklocmm_image_fe_dim}'
            self.args.exp_name += f'__pc{self.args.minklocmm_cloud_fe_dim}'
            self.args.exp_name += f'__{self.args.minklocmm_cloud_planes}'
            self.args.exp_name += f'__{self.args.minklocmm_cloud_layers}'
            self.args.exp_name += f'__{self.args.minklocmm_cloud_topdown}'
            self.args.exp_name += f'__allstg{str(self.args.minklocmm_image_useallstages)[:1]}'
        elif self.args.model in ['minkloconly']:
            # self.args.exp_name += f'++minkloc'
            self.args.exp_name += f'__{self.args.teacher_minkloconly_planes}'
            self.args.exp_name += f'__{self.args.teacher_minkloconly_layers}'
            self.args.exp_name += f'__{self.args.teacher_minkloconly_topdown}'
            self.args.exp_name += f'__{self.args.teacher_minkloconly_dim}'
        elif self.args.model in ['adafusionv2']:
            # self.args.exp_name += f'++adav2'
            self.args.exp_name += f'__{self.args.teacher_adav2_image_fe}'
            self.args.exp_name += f'__img{self.args.teacher_adav2_image_fe_dim}'
            self.args.exp_name += f'__pc{self.args.teacher_adav2_cloud_fe_dim}'
            self.args.exp_name += f'__{self.args.teacher_adav2_cloud_planes}'
            self.args.exp_name += f'__{self.args.teacher_adav2_cloud_layers}'
            self.args.exp_name += f'__{self.args.teacher_adav2_cloud_topdown}'
            self.args.exp_name += f'__allstg{str(self.args.teacher_adav2_image_useallstages)[:1]}'
            self.args.exp_name += f'__res{str(self.args.teacher_adav2_useattenres)[:1]}'
        elif self.args.model in ['resnet18', 'resnet34', 'resnet50', 'convnext_tiny']:
            self.args.exp_name += f'__{self.args.teacher_image_fe_dim}'
        else:
            raise Exception



        # ---- student model
        if self.args.train_mode == 'train_student':
            self.args.exp_name += f'__S:{self.args.student_model}'

            if self.args.student_model in ['resnet18','resnet34']:
                self.args.exp_name += f'__{self.args.student_output_dim}'

            if self.args.student_model == 'minkloconly':
                self.args.exp_name += f'_{self.args.student_output_dim}'
                self.args.exp_name += f'_{self.args.student_minkloconly_planes}'
                self.args.exp_name += f'_{self.args.student_minkloconly_layers}'
                self.args.exp_name += f'_{self.args.student_minkloconly_topdown}'
        



        assert self.args.train_batch_size == self.args.train_batch_split_size
        self.args.exp_name += f'__b{self.args.train_batch_size}'
        if self.args.train_mode == 'train_student':

            if self.args.distil_method == 'nokd':
                self.args.exp_name += f'__nokd'


            elif self.args.distil_method == 'ours':

                self.args.exp_name += f'__+crodist'
                self.args.exp_name += f'_st2ss{self.args.crosslogitdistloss_weight_st2ss}'

                self.args.exp_name += f'__+crosim'
                self.args.exp_name += f'_st2ss{self.args.crosslogitsimloss_weight_st2ss}'

                self.args.exp_name += f'__+crogdist'
                self.args.exp_name += f'_st2ss{self.args.crosslogitgeodistloss_weight_st2ss}'




            elif self.args.distil_method == 'ours_rkdg':

                self.args.exp_name += f'__+rkdg'
                self.args.exp_name += f'__w{self.args.rkdgloss_weight}'

                self.args.exp_name += f'__+crodist'
                self.args.exp_name += f'_st2ss{self.args.crosslogitdistloss_weight_st2ss}'

                self.args.exp_name += f'__+crosim'
                self.args.exp_name += f'_st2ss{self.args.crosslogitsimloss_weight_st2ss}'

                self.args.exp_name += f'__+crogdist'
                self.args.exp_name += f'_st2ss{self.args.crosslogitgeodistloss_weight_st2ss}'


                

            else:
                raise NotImplementedError

        elif self.args.train_mode == 'train_teacher':
            self.args.exp_name += f'__trainteacher'

        else:
            raise NotImplementedError



        expr_dir = os.path.join(self.args.logdir, self.args.exp_name)
        self.args.models_dir = os.path.join(expr_dir, self.args.models_dir)
        self.args.others_dir = os.path.join(expr_dir, 'others')
        mkdirs([
            self.args.logdir, 
            expr_dir, 
            self.args.models_dir,
            self.args.others_dir,
            ])






        
        os.environ['CUDA_VISIBLE_DEVICES'] = str(self.args.cuda)
        os.environ['OMP_NUM_THREADS'] = str(self.args.num_workers//2)




        return self.args





def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            if not os.path.exists(path):
                os.mkdir(path)
    else:
        if not os.path.exists(path):
            os.mkdir(paths)




if __name__ == '__main__':
    args = Options().parse()