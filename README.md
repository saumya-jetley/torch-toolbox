## Adversarial example generator

This script generates adversarial examples for convolutional neural networks
using fast gradient sign method` presented in `Explaining and harnessing
adversarial examples` (Goodfellow et al. 2015).

## Under-construction

The script can be provided the following inputs:
- mode: preproc: Path to the image folder which contains images for generating adversarial examples on. Things to check inside the code for this:
	-- Size of the input image which needs to match the input layer of the architecture
	-- Mean (for subtraction)
	-- Standard deviation (for division)
	-- Path to the ground truth labels (if not available with the images)
- mode: unproc: Path to the t7 file containing images and labels (both)
- Path of the trained model

###Dependency

This script requires trained [OverFeat](https://github.com/sermanet/OverFeat) network.
Running the `example.lua` or the snippet will automatically create the model
and download other files.

```bash
git clone https://github.com/jhjin/overfeat-torch
cd overfeat-torch
. install.sh
th run.lua
mv model.t7 bee.jpg overfeat_label.lua ..
cd ..
```


### Example

The example script predicts the output category of original and its adversarial examples.

```bash
th example.lua
```

![](example.png)
