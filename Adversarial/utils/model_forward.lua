require 'nn'
require 'nngraph'
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
                local y_hat = model[4]:forward(att_con[2]) 
		local all_outputs = {}
		all_outputs[1]=lfeat
		all_outputs[2]=gfeat
		all_outputs[3]=att_con
		all_outputs[4]=y_hat
		return all_outputs
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
		return all_outputs
	elseif atten_code==3 then    
	end
end
return model_forward
