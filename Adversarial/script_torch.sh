:<<'END'
# Test for CUBS-200 - basic
export action='evaluate'
export mode='preproc'
export path_model='{"#models/cubs-basic/model.net"}'
export atten=0
export batch_size=10
export image_size=80
export noise_intensity=1
export path_save='adv_images'
export path_img='#dataset/cubs-200.t7'
#export path_label='#dataset/label_gt.lua'
#export path_img='#dataset/image_gt.lua'
#export list_labels='#dataset/overfeat_label.lua'
export mean=0 # global mean used to train the network
export std=1  # global std used to train the network
export gpumode=1
export gpusetdevice=1
export platformtype='cuda'
th ./main.lua | tee runtimerecord.txt
#-------results for fwd run---------#
#Total images evaluated:5794	
#Total incorrect predictions:2017	
#Percentage Error:34.811874352779%
END


:<<'END'
#Test for CUBS-200 - basic
export action='generate'
export mode='preproc'
export path_model='{"#models/cubs-basic/model.net"}'
export atten=0
export batch_size=2
export image_size=80
export noise_intensity=8
export path_save='#dataset/cubs-adv'
export path_img='#dataset/cubs-200.t7'
#export path_label='#dataset/label_gt.lua'
#export path_img='#dataset/image_gt.lua'
#export list_labels='#dataset/overfeat_label.lua'
export mean=0 # global mean used to train the network
export std=1  # global std used to train the network
export gpumode=1
export gpusetdevice=1
export platformtype='cuda'
th ./main.lua | tee runtimerecord.txt
END

:<<'END'
#Test for CUBS-200 - basic
export action="evaluate"
export mode='unproc'
export path_model='{"#models/cubs-basic/model.net"}'
export atten=0
export batch_size=2
export image_size=80
export noise_intensity=0
export path_save='#dataset/cubs-adv'
#export path_img='#dataset/cubs-200.t7'
export path_label='#dataset/cubs-adv/label_gt.lua'
export path_img='#dataset/cubs-adv/image_gt.lua'
#export list_labels='#dataset/overfeat_label.lua'
export mean=0 # global mean used to train the network
export std=1  # global std used to train the network
export gpumode=1
export gpusetdevice=1
export platformtype='cuda'
th ./main.lua | tee runtimerecord.txt
END


:<<'END'
# Test for CUBS-200 - 2 level attention
export action='evaluate'
export mode='preproc'
export path_model='{"#models/cubs-2level-1global/mlocal_1.net","#models/cubs-2level-1global/mlocal_2.net","#models/cubs-2level-1global/mglobal_2.net","#models/cubs-2level-1global/matten_1.net","#models/cubs-2level-1global/matten_2.net","#models/cubs-2level-1global/mmatch.net"}'
export atten=2
export batch_size=2
export image_size=80
export noise_intensity=0
export path_save='#dataset/cubs-adv'
export path_img='#dataset/cubs-200.t7'
#export path_label='#dataset/label_gt.lua'
#export path_img='#dataset/image_gt.lua'
#export list_labels='#dataset/overfeat_label.lua'
export mean=0 # global mean used to train the network
export std=1  # global std used to train the network
export gpumode=1
export gpusetdevice=1
export platformtype='cuda'
th ./main.lua | tee runtimerecord.txt
#-------results for fwd run--------#
#Total images evaluated:5794	
#Total incorrect predictions:1557	
#Percentage Error:26.872626855368%
END

:<<'END'
# Test for CUBS-200 - 2 level attention
export action='generate'
export mode='preproc'
export path_model='{"#models/cubs-2level-1global/mlocal_1.net","#models/cubs-2level-1global/mlocal_2.net","#models/cubs-2level-1global/mglobal_2.net","#models/cubs-2level-1global/matten_1.net","#models/cubs-2level-1global/matten_2.net","#models/cubs-2level-1global/mmatch.net"}'
export atten=2
export batch_size=2
export image_size=80
export noise_intensity=8
export path_save='#dataset/cubs-adv'
export path_img='#dataset/cubs-200.t7'
#export path_label='#dataset/label_gt.lua'
#export path_img='#dataset/image_gt.lua'
#export list_labels='#dataset/overfeat_label.lua'
export mean=0 # global mean used to train the network
export std=1  # global std used to train the network
export gpumode=1
export gpusetdevice=1
export platformtype='cuda'
th ./main.lua | tee runtimerecord.txt
END

:<<'END'
# Test for CUBS-200 - basic
export action="evaluate"
export mode='unproc'
export path_model='{"#models/cubs-2level-1global/mlocal_1.net","#models/cubs-2level-1global/mlocal_2.net","#models/cubs-2level-1global/mglobal_2.net","#models/cubs-2level-1global/matten_1.net","#models/cubs-2level-1global/matten_2.net","#models/cubs-2level-1global/mmatch.net"}'
export atten=2
export batch_size=2
export image_size=80
export noise_intensity=0
export path_save='#dataset/cubs-adv'
#export path_img='#dataset/cubs-200.t7'
export path_label='#dataset/cubs-adv/label_gt.lua'
export path_img='#dataset/cubs-adv/image_gt.lua'
#export list_labels='#dataset/overfeat_label.lua'
export mean=0 # global mean used to train the network
export std=1  # global std used to train the network
export gpumode=1
export gpusetdevice=1
export platformtype='cuda'
th ./main.lua | tee runtimerecord.txt
END

:<<'END'
# Test for Imagenet
export action='evaluate'
export mode='unproc'
export path_model='{"#models/imagenet-vgg19/VGG_ILSVRC_19_layers_deploy.prototxt","#models/imagenet-vgg19/VGG_ILSVRC_19_layers.caffemodel"}'
export type_model='caffe'
export atten=0
export batch_size=50
export image_size=224
export norm_range=255
export noise_intensity=0
export path_save='#dataset/imagenet-adv'
#export path_img='#dataset/cubs-200.t7'
export path_label='#dataset/imagenet/label_valgt.lua'
export path_img='#dataset/imagenet/image_valgt.lua'
#export list_labels='#dataset/overfeat_label.lua'
export mean='{123.68 , 116.779, 103.939}' # global mean used to train the network
export std='{1,1,1}'  # global std used to train the network
export gpumode=1
export gpusetdevice=1
export platformtype='cuda'
th ./main.lua | tee runtimerecord.txt
#-------results for fwd run--------#
END

:<<'END'
# Test for Imagenet
export action='evaluate'
export mode='unproc'
export path_model='{"#models/imagenet-vgg19/VGG_ILSVRC_19_layers.net"}'
export type_model='torch'
export atten=0
export batch_size=50
export image_size=224
export norm_range=255
export noise_intensity=0
export path_save='#dataset/imagenet-adv'
#export path_img='#dataset/cubs-200.t7'
export path_label='#dataset/imagenet/label_valgt.lua'
export path_img='#dataset/imagenet/image_valgt.lua'
#export list_labels='#dataset/overfeat_label.lua'
export mean='{123.68 , 116.779, 103.939}' # global mean used to train the network
export std='{1,1,1}'  # global std used to train the network
export gpumode=1
export gpusetdevice=1
export platformtype='cuda'
th ./main.lua | tee runtimerecord.txt
#-------results for fwd run--------#
END

# Test for Imagenet
export action='evaluate'
export mode='unproc'
export path_model='{"#models/imagenet-vgg19/VGG_ILSVRC_19_layers_224conv.net"}'
export type_model='torch'
export atten=0
export batch_size=25
export image_size=224
export norm_range=255
export noise_intensity=0
export path_save='#dataset/imagenet-adv'
#export path_img='#dataset/cubs-200.t7'
export path_label='#dataset/imagenet/label_valgt.lua'
export path_img='#dataset/imagenet/image_valgt.lua'
#export list_labels='#dataset/overfeat_label.lua'
export mean='{123.68 , 116.779, 103.939}' # global mean used to train the network
export std='{1,1,1}'  # global std used to train the network
export gpumode=1
export gpusetdevice=1
export platformtype='cuda'
th ./main.lua | tee runtimerecord.txt
#-------results for fwd run--------#

