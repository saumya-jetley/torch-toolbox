require 'image'
require 'torch'

local function save_batch(imgs,labels,save_id,batch_size,path_save)
	for ind=1,batch_size,1 do
		image.save(save_id..'_'..labels[ind]..'.png', imgs[ind])
		save_id = save_id + 1
	end
	return save_id
end
return save_batch
