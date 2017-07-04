require 'nn'
require 'cunn'
require 'nngraph'
function model_load(names, type_model, atten_code, cast)
	if atten_code==0 then
			if type_model=='caffe' then
				require 'loadcaffe'
				model_wts = loadcaffe.load(names[1],names[2])
				model_wts:remove(#model_wts) --removing the softmax layer at the end for consistency with code
				model_wts:remove(44) --removing dp1 (**sj)
				model_wts:remove(41) --removing dp2 (**sj)
			else
				model_wts = torch.load(names[1])
			end
			model = nn.Sequential():add(nn.Copy('torch.DoubleTensor', torch.type(cast(torch.Tensor())))):add(cast(model_wts))
		return model
	elseif atten_code==1 then
                model_wts_local = torch.load(names[1])
                model_local = nn.Sequential()
                model_local:add(nn.Copy('torch.DoubleTensor', torch.type(cast(torch.Tensor()))))
                model_local:add(cast(model_wts_local))

                model_wts_global = torch.load(names[2])
                model_global = nn.Sequential()
                model_global:add(cast(model_wts_global))

                model_wts_atten = torch.load(names[3])
                model_atten = nn.Sequential()
                model_atten:add(cast(model_wts_atten))

                model_wts_match = torch.load(names[4])
                model_match = nn.Sequential()
                model_match:add(cast(model_wts_match))
		
		model = {}
		model[1] = model_local
		model[2] = model_global
		model[3] = model_atten
		model[4] = model_match
		return model
	elseif atten_code==2 then
                model_wts_local1 = torch.load(names[1])
                model_local1 = nn.Sequential()
                model_local1:add(nn.Copy('torch.DoubleTensor', torch.type(cast(torch.Tensor()))))
                model_local1:add(cast(model_wts_local1))

                model_wts_local2 = torch.load(names[2])
                model_local2 = nn.Sequential()
                model_local2:add(cast(model_wts_local2))

                model_wts_global2 = torch.load(names[3])
                model_global2 = nn.Sequential()
                model_global2:add(cast(model_wts_global2))

                model_wts_atten1 = torch.load(names[4])
                model_atten1 = nn.Sequential()
                model_atten1:add(cast(model_wts_atten1))

                model_wts_atten2 = torch.load(names[5])
                model_atten2 = nn.Sequential()
                model_atten2:add(cast(model_wts_atten2))

                model_wts_match = torch.load(names[6])
                model_match = nn.Sequential()
                model_match:add(cast(model_wts_match))

                model = {}
                model[1] = model_local1
		model[2] = model_local2 
                model[3] = model_global2
                model[4] = model_atten1
		model[5] = model_atten2
                model[6] = model_match
                return model
	elseif atten_code==3 then
	end
end
return model_load
