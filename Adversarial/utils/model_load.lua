function model_load(names, atten_code, cast)
	if atten_code==0 then
		model_wts = torch.load(names[1])
		model = nn.Sequential():add(nn.Copy('torch.DoubleTensor', torch.type(cast(torch.Tensor())))):add(cast(model_wts))
		return model
	elseif atten_code==1 then
	elseif atten_code==2 then
	elseif atten_code==3 then
	end
end
return model_load
