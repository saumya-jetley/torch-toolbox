function model_forward(model, atten_code, x)
	if atten_code==0 then
		local y_hat = model:forward(x)
		return y_hat
	elseif atten_code==1 then
		
	elseif atten_code==3 then
	end
end
return model_forward
