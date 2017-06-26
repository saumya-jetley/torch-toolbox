require 'torch'
require 'image'

torch.setdefaulttensortype('torch.FloatTensor')

local function preprocess_data(image_list, label_list, batch_size, image_size, mean, std)
	--create the mean image for subtraction
	if type(mean)=='number' then
		mean_vector=torch.Tensor(3):fill(mean)
	end
	mean_image = torch.repeatTensor(mean_vector, image_size, image_size, 1)
	if type(std)=='number' then
		std_vector=torch.Tensor(3):fill(std)
	end
	std_image = torch.repeatTensor(std_vector, image_size, image_size, 1)
	-- create the 4D image tensor and 2D label tensor
	image_tensor= torch.Tensor(batch_size,3,image_size, image_size):fill(0)
	label_tensor = torch.Tensor(batch_size):fill(0)
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
		local proc_image = sq_image:permute(2,3,1):add(-mean):div(std):permute(3,1,2)
		image_tensor[i]:copy(proc_image)
		label_tensor[i] = label_nlist[i]
	end
	return image_tensor, label_tensor
end
return preprocess_data
