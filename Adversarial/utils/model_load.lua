function model_load(names_list, atten_code, cast)
	names = loadstring('return'.. names_list)()
	if atten_code==0 then
		model_wts = torch.load(names[1])
		model = nn.Sequential():add(nn.Copy('torch.FloatTensor', torch.type(cast(torch.Tensor())))):add(model_wts)
	elseif atten_code==1 then
	elseif atten_code==2 then
	elseif atten_code==3 then
	end
end
return model_load
