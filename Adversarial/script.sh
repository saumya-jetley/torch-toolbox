cmd_params={
action = 'evaluate', -- 'generate'
mode = 'unproc', -- 'preproc'
path_model = '#overfeat-torch/model.net',
batch_size = 1,
image_size = 231,        -- small net requires 231x231
noise_intensity = 1,           -- pixel intensity for gradient sign
save_image = 'adv_images',
--path_img = 'data.t7'
path_label = '#dataset/label_gt.lua', -- label file (in order*)
path_img = '#dataset/image_gt.lua', -- image file (in order*)
mean = 118.380948/255,   -- global mean used to train overfeat
std = 61.896913/255,     -- global std used to train overfeat
list_labels = '#dataset/overfeat_label.lua'
}

