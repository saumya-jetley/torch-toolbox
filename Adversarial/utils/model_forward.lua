require 'nn'
require 'nngraph'
require 'image'
function model_forward(model, atten_code, x)
	if atten_code==0 then
		local y_hat = model:forward(x)
		local all_outputs={}
		all_outputs[1]=y_hat
		return all_outputs
	elseif atten_code==1 then
		local lfeat = model[1]:forward(x)
                local gfeat = model[2]:forward(lfeat)
                local att_con = model[3]:forward({lfeat,gfeat})
		-- the attention mask is the first element of the table
		print(att_con[1][1]:size())
		image.save('att_img.png', att_con[1][1]:clone():clamp(0,1))
		return
	--[[
                local y_hat = model[4]:forward(att_con[2]) 
		local all_outputs = {}
		all_outputs[1]=lfeat
		all_outputs[2]=gfeat
		all_outputs[3]=att_con
		all_outputs[4]=y_hat
		return all_outputs
	]]--	
	elseif atten_code==2 then
		local lfeat_1 = model[1]:forward(x)           
		local lfeat_2 = model[2]:forward(lfeat_1)           
		local gfeat_2 = model[3]:forward(lfeat_2)
		local att_con_1 = model[4]:forward({lfeat_1,gfeat_2})
		local att_con_2 = model[5]:forward({lfeat_2,gfeat_2})
		local y_hat = model[6]:forward({att_con_1[2], att_con_2[2]})         
		local all_outputs = {}
		all_outputs[1]=lfeat_1
		all_outputs[2]=lfeat_2
		all_outputs[3]=gfeat_2
		all_outputs[4]=att_con_1
		all_outputs[5]=att_con_2
		all_outputs[6]=y_hat
		print(att_con_1[1][1]:repeatTensor(3,1,1):size())
		print(att_con_1[1][1]:repeatTensor(3,1,1):max(1))
		att_image_1 = att_con_1[1][1]:clone():mul(10):clamp(0,1):repeatTensor(3,1,1)
		att_image_2 = att_con_2[1][1]:clone():mul(10):clamp(0,1):repeatTensor(3,1,1)
		image.save('att_img_1.png',image.scale(att_image_1:double(),80,80))
		image.save('att_img_2.png',image.scale(att_image_2:double(),80,80))
		return
--		return all_outputs
	elseif atten_code==3 then    
	end
end
return model_forward
