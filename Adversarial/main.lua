#!/usr/bin/env th
require('xlua')
require('torch')
require('nn')
require('image')
require('paths')

adversarial_fast = require'utils/adversarial_fast'
preprocess_data = require 'utils/preprocess_data'
unprocess_data = require 'utils/unprocess_data' 
save_batch = require 'utils/save_batch'

torch.setdefaulttensortype('torch.FloatTensor')

cmd_params={
action = 'generate', -- 'evaluate'
mode = 'unproc', -- 'preproc'
path_model = 'overfeat-torch/model.net',
batch_size = 1,
image_size = 231,        -- small net requires 231x231
noise_intensity = 1,           -- pixel intensity for gradient sign
save_image = 'adv_images',
--path_img = 'data.t7'
path_label = 'dataset/label_gt.lua', -- label file (in order*)
path_img = 'dataset/image_gt.lua', -- image file (in order*)
mean = 118.380948/255,   -- global mean used to train overfeat
std = 61.896913/255,     -- global std used to train overfeat
list_labels = 'dataset/overfeat_label.lua'
}

-- update the cmd_params from command terminal input
cmd_params = xlua.envparams(cmd_params)

mode = cmd_params.mode
path_model = cmd_params.path_model
batch_size = cmd_params.batch_size
image_size = cmd_params.image_size
noise_intensity = cmd_params.noise_intensity
save_image = cmd_params.save_image
--path_img = cmd_params.path_img
path_label = cmd_params.path_label
path_img = cmd_params.path_img
mean = cmd_params.mean
std = cmd_params.std
list_labels = cmd_params.list_labels
action = cmd_params.action

tot_incorrect = torch.Tensor(1):fill(0)
tot_evals = torch.Tensor(1):fill(0)
save_id = 0

if not paths.filep(list_labels) then
	print('List of label names file not found!')
else
	ll = require(list_labels)
end

-- Get the image and labels (in tensor)
if mode=='preproc' then
	if not paths.filep(path_img) then
		print('database file (t7) not found!')
	else
		data = torch.load(path_img)
		images = data.train_data --sj
		labels = data.train_labels --sj
		num_img = images:size(1) --sj
	end
elseif mode=='unproc' then
	if not paths.filep(path_img) or not paths.filep(path_label) then
		print('Either image folder or label file not found!')
	else
		images = require(path_img)
		labels = require(path_label)
		num_img = #images
	end
end

-- Get the model
if not paths.filep(path_model) then
  print('model not found!') 
else
  model = torch.load(path_model) --sj
end

-- randomize the images indices for access
local shuffled_indices = torch.randperm(num_img):long()
local batch_indices = shuffled_indices:split(batch_size)

if mode=='unproc' then
	--shuffle the elements in the tables
	for sh_ind = 1,1,num_img do
		images[sh_ind],images[shuffled_indices[sh_ind]] = images[shuffled_indices[sh_ind]], images[sh_ind]
		labels[sh_ind],labels[shuffled_indices[sh_ind]] = labels[shuffled_indices[sh_ind]], labels[sh_ind]
	end	
end

for ind, ind_batch in ipairs(batch_indices) do
	--DATA
	-- create a NEW 4D tensor for images/NEW 2D tensor for labels
	if mode=='preproc' then
		input_imgs = images:index(1,v)
		input_lbs = labels:index(1,v)
	elseif mode=='unproc' then
		--select the batch_sized subsets using unpack
		input_imgs, input_lbs = preprocess_data(unpack(images,(ind-1)*batch_size+1,ind*batch_size), unpack(labels,(ind-1)*batch_size+1,ind*batch_size), batch_size, image_size, mean, std)
	end  
	--OPERATION
	if action=='generate' then --generate 'adversarial examples'
		-- get trained model (switch softmax to logsoftmax) & set loss
		model.modules[#model.modules] = nn.LogSoftMax()
		local loss = nn.ClassNLLCriterion()
		-- call the generator
		print(input_lbs)

		local img_adv = adversarial_fast(model, loss, input_imgs:clone(), input_lbs, std, noise_intensity)
		model.modules[#model.modules] = nn.SoftMax()

		--[[
		-- check prediction results
		local pred = model:forward(input_imgs)
		local val, idx = pred:max(pred:dim())
		print('==> original:', ll[ idx[1] ], 'confidence:', val[1])

		local pred = model:forward(img_adv)
		local val, idx = pred:max(pred:dim())
		print('==> adversarial:', ll[ idx[1] ], 'confidence:', val[1])

		local img_diff = torch.add(input_imgs, -img_adv)
		print('==> mean absolute diff between the original and adversarial images[min/max]:', torch.abs(img_diff):mean())

		image.save('img.png', input_imgs[1]:mul(std):add(mean):clamp(0,255))
		image.save('img_adv.png', img_adv[1]:mul(std):add(mean):clamp(0,255))
		image.save('img_diff.png',img_diff[1]:mul(std):mul(255):clamp(0,255))
		return
		--]]
		--unnormalise the adversarial images
		local img_adv_normal = unprocess_data(img_adv, batch_size, image_size, mean, std)
		-- save the images in the save_folder
		save_id = save_batch(img_adv_normal, save_id, batch_size)

				
	elseif action=='evaluate' then -- evaluate the accuracy
		--forward pass/ get prediction
		local y_hat = model:forward(input_imgs)
		local val, idx = y_hat:max(2) --sj		
		--compare with the ground truth
		local incorrect = torch.ne(idx, input_lbs)
		--accumulate the error
		tot_incorrect = tot_incorrect:add(incorrect:sum())
		tot_evals = tot_evals:add(incorrect:size(1))
	end
	--]]
end

if action=='evaluate' then
	-- print the cumulative error
	print('Total images evaluated:'.. tot_evals[1])
	print('Total incorrect predictions:'.. tot_incorrect[1])
	print('Percentage Error:'.. tot_incorrect:div(tot_evals):mul(100) .. '%')
end

print('Succesfully Completed The Code Run')
--[[
if pcall(require,'qt') then
  local img_cat = torch.cat(torch.cat(img, img_adv, 3), img_diff:mul(127), 3)
  image.display(img_cat)
end
--]]

