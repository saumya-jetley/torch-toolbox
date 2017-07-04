# Test for CUBS-200 - for attention
export action="evaluate"
export mode='unproc'
export path_model='{"#models/cubs-2level-1global/mlocal_1.net","#models/cubs-2level-1global/mlocal_2.net","#models/cubs-2level-1global/mglobal_2.net","#models/cubs-2level-1global/matten_1.net","#models/cubs-2level-1global/matten_2.net","#models/cubs-2level-1global/mmatch.net"}'
export atten=2
export batch_size=2
export image_size=80
export noise_intensity=8
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

