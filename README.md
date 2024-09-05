Windows installation steps for DeepForest:

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
