require('nn')
require('cunn')
torch.setdefaulttensortype('torch.DoubleTensor')
model_forward=require('utils/model_forward')
model_backward=require('utils/model_backward')

-- "Explaining and harnessing adversarial examples"
-- Ian Goodfellow, 2015
local function adversarial_fast(model, loss, x, y, std, intensity, cast, atten)
   assert(loss.__typename == 'nn.CrossEntropyCriterion')
   local intensity = intensity or 1

   -- consider x as batch
   local batch = false
   if x:dim() == 3 then
      print('Dimension 3')
      x = x:view(1, x:size(1), x:size(2), x:size(3))
      batch = true
   else
      print('Dimension 4')
      batch = true
   end

   -- consider y as tensor
   if type(y) == 'number' then
      y = torch.Tensor({y}):typeAs(x)
   end

   -- compute output
   --local y_hat = model:forward(x)
   local outputs  = model_forward(model, atten, x)	
   local y_hat = outputs[#outputs]

   -- use predication as label if not provided
   local _, target = nil, y
   if target == nil then
      print('still going here is wrong')
      _, target = y_hat:max(y_hat:dim())
   end

   -- find gradient of input (inplace)
   local cost = loss:forward(y_hat, target)
   local cost_grad = loss:backward(y_hat, target)
   --local x_grad = model:backward(x, cost_grad)
   outputs[#outputs+1]=cost_grad
   local x_grad = model_backward(model, atten, x, outputs)
   local noise = (x_grad:sign():mul(intensity/255)):double()


   -- normalize noise intensity
   if type(std) == 'number' then
      noise:div(std)
   else
      for c = 1, 3 do
         noise[{{},{c},{},{}}]:div(std[c])
      end
   end

   -- return adversarial examples (inplace)
   return x:add(noise)
end

return adversarial_fast
