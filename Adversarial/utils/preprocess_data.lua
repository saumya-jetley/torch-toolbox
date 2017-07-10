require 'torch'
require 'image'

torch.setdefaulttensortype('torch.FloatTensor')

local function preprocess_data(image_list, label_list, batch_size, image_size, mean, std, norm_range)
	--create the mean image for subtraction
	local mean = loadstring('return'.. mean)()
	local mean_tensor = torch.Tensor(mean)
	local std = loadstring('return'.. std)()
	local std_tensor = torch.Tensor(std)
	local len = mean_tensor:size(1)
	if len==1 then
		mean_tensor=torch.Tensor(3):fill(mean)
		std_tensor=torch.Tensor(3):fill(std)
	end
	local mean_image = torch.repeatTensor(mean_tensor, image_size, image_size, 1)
	local std_image = torch.repeatTensor(std_tensor, image_size, image_size, 1)
	--if type(std)=='number' then
	--	std_vector=torch.Tensor(3):fill(std)
	--end
	
	-- create the 4D image tensor and 2D label tensor
	local image_tensor= torch.Tensor(batch_size,3,image_size, image_size):fill(0)
	local label_tensor = torch.Tensor(batch_size):fill(0)
	-- populate the tensors
	if batch_size==1 then
		image_nlist = {}
		label_nlist = {}
		image_nlist[1]=image_list
		label_nlist[1]=label_list
	else
		image_nlist = image_list
		label_nlist = label_list
	end
	for i=1,batch_size,1 do
		local raw_image = image.scale(image.load(image_nlist[i], 3, 'double'),'^'..image_size)
		local sub_h = math.floor((raw_image:size(2)-image_size)/2)+1
		local sub_w = math.floor((raw_image:size(3)-image_size)/2)+1
		local sq_image = raw_image[{{},{sub_h,sub_h+image_size-1},{sub_w,sub_w+image_size-1}}]
		local proc_image = sq_image:permute(2,3,1):mul(norm_range):add(-mean_image):cdiv(std_image):permute(3,1,2)
		image_tensor[i]:copy(proc_image)
		label_tensor[i] = label_nlist[i]
	end
	return image_tensor, label_tensor
end
return preprocess_data
