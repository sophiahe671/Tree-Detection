<h1 align="center">
  DeepForest
</h1>

![](https://deepforest.readthedocs.io/en/v1.3.3/_images/getting_started1.png)

## Mission
As part of the ever changing landscapes and dangers of climate change, estimates of above ground biomass are needed to monitor how different areas change over time. To estimate the above ground biomass, DeepForest detects and segments the tree canopy's coverage, which should correlate with the amount of stored carbon in the vegetation. My mission, along with two other engineers, was to optimize and fine-tune this model by tuning the patch size, acceptance threshold, and other hyperparameters.

## Summary
DeepForest is a Python library that detects and draws boxes around specific objects, primarily trees and birds. It has pretrained models for predicting trees, although there are ways to fine tune those models if you supply more data.

The ‘default’ function for predicting trees in an image is predict_image. Here is an example script that uses it:
``` Ruby
from deepforest import main
from deepforest import get_data
import matplotlib.pyplot as plt

model = main.deepforest()
model.use_release()

sample_image_path = get_data("OSBS_029.png")
img = model.predict_image(path=sample_image_path, return_plot=True)


#predict_image returns plot in BlueGreenRed (opencv style), but matplotlib
likes RedGreenBlue, switch the channel order. Many functions in deepforest
will automatically perform this flip for you and give a warning.
plt.imshow(img[:,:,::-1])
```
The DeepForest library was used as a baseline for further tuning to increase the accuracy of the model.

## Tech Stack
- **Python**: Backend development for running the models and generating tree image predictions

## Windows Installations Steps for DeepForest
Install conda (run the following commands in Powershell)

$ curl https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe -o miniconda.exe  
$ start miniconda.exe  
$ del miniconda.exe  

Navigate to the Anaconda Powershell Prompt app on your computer  
On the Anaconda Powershell Prompt, install and set up DeepForest (run the following commands)  

$ pip install DeepForest  
	$ conda create -n deepforest python=3 pytorch torchvision -c pytorch  
	$ conda activate deepforest  
	$ conda install deepforest -c conda-forge  
 
Running a DeepForest Script on Windows:  

Open the Anaconda Powershell Prompt  
Run the following command:  
$ conda activate deepforest  

You need to run this command every time you open up the Anaconda Powershell Prompt  
Make sure all the files the script needs is in this folder:  
<path to your miniconda3 folder> /miniconda3/envs/deepforest/Lib/site-packages/deepforest/data  

If you don’t know where your folder is, run the script with whatever data it has, and it should show up in the error message  
Your deepforest_config.yml file should also be in here if you want to edit it  

Run the python script from the Anaconda prompt  

## Challenges
- **Time to Annotate**: Annotating a large enough set of images to effectively train a model is time intensive. Needs to be done by hand and be accurate

## What I learned
- Evaluation metrics to measure the accuracy vs precision of the model require determining IoU (intersection over union) and F1 score accurately
- Models require different hyperparameters depending on the setting of the image (rurual vs. urban)
- Annotating ground-truth images requires high precision because this will influence the generation of the model

## Summary
These findings are helpful to determine above ground biomass and CO2 absorption by trees. The predictions from both models can be used to calculate the total area with canopy coverage. Fast growing trees absorb CO2 faster than older trees which can be determined by the size and type of tree. The flexibility of DeepForest allow expandability and ability to visualize and calculate the square footage of canopy cover.
