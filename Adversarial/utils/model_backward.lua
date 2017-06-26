function model_backward(model, atten_code, x, cost_grad)
	if atten_code==0 then
		local x_grad = model:backward(x, cost_grad)
		return x_grad
	elseif atten_code==1 then
	elseif atten_code==3 then
	end
end
return model_backward
