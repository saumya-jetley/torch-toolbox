require 'torch'
local aug_utils = {}
cparams = _G.value


-- support function - Data casting
function aug_utils.cast(t)
   if cparams.platformtype == 'cuda' then
      return t:cuda()
   elseif cparams.platformtype == 'double' then
      return t:double()
   --elseif cparams.platformtype == 'cl' then
   --  require 'clnn'
   --   return t:cl()
   else
      error('Unknown type '..cparams.platformtype)
   end
end

-- support function - Data augmentation
function aug_utils.hflip_aug(input)
      --hflip
      local bs = input:size(1)
      local flip_mask = torch.randperm(bs):le(bs/2)
      for i=1,input:size(1) do
        if flip_mask[i] == 1 then image.hflip(input[i], input[i]) end
      end
    return input
end

function aug_utils.rot_aug(input, pad)
    -- Rotation
    assert(input:dim() == 4)
    local imsize = input:size(4)
    local padded = nn.SpatialZeroPadding(pad,pad,pad,pad):forward(input)
    local x = torch.random(1,pad*2 + 1)
    local y = torch.random(1,pad*2 + 1)
    local input_rot = padded:narrow(4,x,imsize):narrow(3,y,imsize)
    return input_rot:contiguous()
end

return aug_utils
