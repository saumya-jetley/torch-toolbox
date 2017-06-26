require 'torch'
require 'image'


local function unprocess_data(output_imgs, batch_size, image_size, mean, std)
        --create the mean image for subtraction
        if type(mean)=='number' then
                mean_vector=torch.Tensor(3):fill(mean)
        end
        local mean_image = torch.repeatTensor(mean_vector, batch_size, image_size, image_size, 1)
        if type(std)=='number' then
                std_vector=torch.Tensor(3):fill(std)
        end
        local std_image = torch.repeatTensor(std_vector, batch_size, image_size, image_size, 1)        
	output_imgs:permute(1,3,4,2):cmul(std_image):add(mean_image):permute(1,4,2,3):clamp(0,255)

	return output_imgs
end
return unprocess_data

