# EO_landcover
Milestone repository for Earth Observation land cover task with CNNs.


First install anaconda on your local/remote machine (accept the Licence Agreement and allow Anaconda to be added to your `PATH`)

```
wget https://repo.anaconda.com/archive/Anaconda3-2020.02-Linux-x86_64.sh
bash Anaconda3-2020.02-Linux-x86_64.sh ---> any other newer version of anaconda can also be used
```

Source the `.bashrc` to load the new `PATH` variable (or logout and login back). Verify that Anaconda is correctly installed and linked by opening Python from the command line. Python from Anaconda should be listed as below

```
$ python
Python 3.7.1 (default, Dec 14 2018, 19:28:38) 
[GCC 7.3.0] :: Anaconda, Inc. on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> 
```


Now that the installation is complete, create an Anaconda environment

`conda create --name eolc python=3.7`


To work with the repository, activate the environment with the command

`conda activate eolc`


Install packages in `requirements.txt` with the command

`pip install -r requirements.txt`


If any basic package is missing at any time (i.e. numpy), it can be easily installed via 

`pip install nameofthepackage`


To deactivate Anaconda environment after working

`conda deactivate`


Download the EuroSAT dataset from here:
`http://madm.dfki.de/files/sentinel/EuroSATallBands.zip`
