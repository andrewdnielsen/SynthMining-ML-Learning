# SynthMining-ML-Learning

## Python Setup

### 1. Install python version 3.11
You can check your current installation of python by running `python --version` or `python3 --version`. If you have an outdated version or you don't have python installed, please make sure to install the latest version at https://www.python.org/downloads/

### 2. Install pip
In order to install and manage python packages, you will need to install pip. Follow the instructions here: https://pip.pypa.io/en/stable/installation/

To verify you have installed pip correctly, try the following command:
```
> python -m pip --version 
pip 23.1.2 from C:\Path\To\pip (python 3.11)
```

Note that when working with different versions of python on a system, you might have to specify which one you are using. You can also create an alias for `python` or `python3` that points to the executable path for your appropriate python installation.

### 3. Setting up a python virtual environment
When working with various python libraries, it is good practice to maintain a virtual environment, which will contain all of the appropriate modules and dependencies.

```
> python -m pip install virtualenv
> python -m venv VIRTUAL-ENVIRONMENT-NAME.venv
```

To activate your virtual environment:
```
Windows:
> .\VIRTUAL-ENVIRONMENT-NAME.venv\Scripts\activate

MacOS / Linux:
> ./VIRTUAL-ENVIRONMENT-NAME.venv/source/activate
```

### 4. Installing required packages
With your virtual environment activated, you can install packages using pip (will be bound to the pip executable in your venv)

```
> pip install -r requirements.txt
```

With this, you should be all set! If you're using VSCode, make sure to select the python executable within your venv directory as your python interpreter so that linting works properly