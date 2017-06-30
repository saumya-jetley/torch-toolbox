require 'nn'
require 'nngraph'
function model_backward(model, atten_code, x, outputs)
	if atten_code==0 then
		local x_grad = model:backward(x, outputs[#outputs])
		return x_grad
	elseif atten_code==1 then
		lfeat=outputs[1]
		gfeat=outputs[2]
		att_con=outputs[3]
		y_hat=outputs[4]
		local df_context = model[4]:backward({att_con[2]}, outputs[#outputs])
                local df_feat = model[3]:backward({lfeat,gfeat}, {torch.rand(outputs[3][1]:size()):cuda():fill(0), df_context})         
                local df_lfeat = model[2]:backward(lfeat, df_feat[2])                  
                local x_grad = model[1]:backward(x,(df_lfeat+df_feat[1])/2)
		return x_grad
	elseif atten_code==2 then
		lfeat_1=outputs[1]
		lfeat_2=outputs[2]
		gfeat_2=outputs[3]
		att_con_1=outputs[4]
		att_con_2=outputs[5]
		y_hat=outputs[6]
		local df_context = model[6]:backward({att_con_1[2], att_con_2[2]}, outputs[#outputs])
		local df_feat_2 = model[5]:backward({lfeat_2,gfeat_2}, {torch.rand(att_con_2[1]:size()):cuda():fill(0), df_context[2]})
		local df_feat_1 = model[4]:backward({lfeat_1,gfeat_2}, {torch.rand(att_con_1[1]:size()):cuda():fill(0), df_context[1]})
		local df_lfeat_2 = model[3]:backward(lfeat_2, (df_feat_1[2]+df_feat_2[2])/2)
		local df_lfeat_chain_1 = model[2]:backward(lfeat_1,(df_lfeat_2+df_feat_2[1])/2)
		local x_grad = model[1]:backward(x,(df_feat_1[1]+df_lfeat_chain_1)/2)	
		return x_grad
	elseif atten_code==3 then
	end
end
return model_backward
