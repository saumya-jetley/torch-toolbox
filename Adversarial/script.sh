# Test for CUBS-200 - basic
export action='generate' #"evaluate"
export mode='preproc'
export path_model='{"#models/cubs-basic/model.net"}'
export atten=0
export batch_size=1
export image_size=80
export noise_intensity=1
export save_image='adv_images'
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

