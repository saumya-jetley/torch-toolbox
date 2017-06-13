#!/usr/bin/env th
require('torch')
require('nn')
require('image')
require('paths')
torch.setdefaulttensortype('torch.FloatTensor')

cmd_params{
mode = preproc
path_model = 'model.net'
path_img = 'data.t7'
batch_size = 20
-- mode = unproc
-- size_image = 231        -- small net requires 231x231
-- path_label = label_file -- label file (in order*)
-- mean = 118.380948/255   -- global mean used to train overfeat
-- std = 61.896913/255     -- global std used to train overfeat
-- intensity = 1           -- pixel intensity for gradient sign
-- path_img = imgname_file -- image file (in order*)
}

-- Get the image and labels (in tensor)
if cmd_params.mode=='preproc' then
	if not paths.filep(cmd_params.path_img) then
		print('database file (t7) not found!')
	else
		data = torch.load(cmd_params.path_img)
		images = data.train_data --sj
		labels = data.train_labels --sj
		num_img = images:size()[1] --sj
	end
elseif mode=='unproc' then
	if not paths.filep(cmd_params.path_img) or not paths.filep(cmd_params.path_label) then
		print('Either image folder or label file not found!')
	else
		images = require(cmd_params.path_img) --sj
		labels = require(cmd_params.path_label) --sj
		num_img = #images --sj
	end
end

-- Get the model
if not paths.filep(cmd_params.path_model) then
  print('model not found!') 
else
  model = torch.load(cmd_params.path_model) --sj
end
-- get trained model (switch softmax to logsoftmax)
model.modules[#model.modules] = nn.LogSoftMax()
-- set loss function
local loss = nn.ClassNLLCriterion()


for img_ind=1:bs:num_img do

  -- create a NEW 4D tensor for the images
  -- create a NEW 2D tensor for the labels
	if cmd_params.action=='generate' then
		-- Generate the adversarial samples
	else if cmd_params.action=='evaluate' then
		-- evaluate the accuracy
	end
end

-- resize input/label
local img = image.scale(image.load(path_img), '^'..eye)
local tx = math.floor((img:size(3)-eye)/2) + 1
local ly = math.floor((img:size(2)-eye)/2) + 1
img = img[{{},{ly,ly+eye-1},{tx,tx+eye-1}}]
img:add(-mean):div(std)

-- generate adversarial examples
local img_adv = require('adversarial-fast')(model, loss, img:clone(), label_nb, std, intensity)

model.modules[#model.modules] = nn.SoftMax()
-- check prediction results
local pred = model:forward(img)
local val, idx = pred:max(pred:dim())
print('==> original:', label[ idx[1] ], 'confidence:', val[1])

local pred = model:forward(img_adv)
local val, idx = pred:max(pred:dim())
print('==> adversarial:', label[ idx[1] ], 'confidence:', val[1])

local img_diff = torch.add(img, -img_adv)
print('==> mean absolute diff between the original and adversarial images[min/max]:', torch.abs(img_diff):mean())

image.save('img.png', img:mul(std):add(mean):clamp(0,255))
image.save('img_adv.png', img_adv:mul(std):add(mean):clamp(0,255))

if pcall(require,'qt') then
  local img_cat = torch.cat(torch.cat(img, img_adv, 3), img_diff:mul(127), 3)
  image.display(img_cat)
end

