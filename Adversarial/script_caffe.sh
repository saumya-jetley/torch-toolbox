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

