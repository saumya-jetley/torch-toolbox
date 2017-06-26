require 'image'
require 'torch'

local function save_batch(imgs, labels, save_id, batch_size, im_file, lb_file, save_path)
	for ind=1,batch_size,1 do
		-- save the images
		image.save(save_path..'/'..save_id..'_'..labels[ind]..'.png', imgs[ind])
		-- write the text file with image_names
		print('\"'..save_path..'/'..save_id..'_'..labels[ind]..'.png\",')
		im_file:write('\n\"'..save_path..'/'..save_id..'_'..labels[ind]..'.png\",')
		-- write the text file with image_labels
		print('\"'..labels[ind]..'\",')
		lb_file:write('\n\"'..labels[ind]..'\",')
		save_id = save_id+1
	end
	return save_id
end
return save_batch
