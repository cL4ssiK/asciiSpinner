## Description
This software is designed to display 3D models on command line using ascii characters as graphics.

At least in theory the program should be able to display any 3D model in .obj format.

I made this project on windows machine and displaying is made in windows command prompt (CMD). I have not implemented cross platform support, but feel free to try on Linux, Mac-OS or other command line environments than CMD.

Since the code is not optimized beyond replacing most looping structures with numpy vectorization operations I have limited the maxium amount of faces model is allowed to have to 10k. If faces exceed 10k, the program will try simplifying the model until it reaches less than 10k faces. If simplification does not reduce arbitarily set percentage from previous attempt, program exits. Also scaling the model is not (yet) implemented, so if model is very small, the details will be overrun. This is becouse of the crude way the projection from 3D floating point coordinates to 2D integer buffer is implemented.
One more issue is the starting position of the model. Currently program scales the buffer size to the model size, but there is no way to easily do the initial rotation of the model. So if the orientation of the original model is weird, so is the orientation it spins. Of course you can hardcode the proper rotation by changing the sourse code.

Motivation for this project was video of 3D ascii donut from internet. Friend of mine sort of challenged me to implement it. I realized that spinning 3D model is basic linear algebra, so i decided to test my math knowledge in practice. And what is cooler than spinning ascii donut? Well... Spinning ascii "anything" :D

## Setup (first time)
Reason why setup requires pyenv is the model simplification property. It contains dependency written in C++ and it does not play well with python versions more recent than 3.11.x.

### 0. Install pyenv
For windows install [pyenv-win](https://github.com/pyenv-win/pyenv-win?tab=readme-ov-file#installation).

I installed via pip since i had python on my system already and it worked fine.

Linux and mac-os support official pyenv. Instructions coming...

### 1. Clone the project
```bash
git clone https://github.com/cL4ssiK/asciiSpinner
```

### 2. Move into project root folder
```bash
cd asciiSpinner
```

### 3. Run the setup script (windows cmd)
```bash
setup.bat
```
Setup script for Linux/Mac-OS coming...

Run the setup script only if environment changes.

## Usage

### 1. Activate virtual environment
In project root run the following:
```bash
.venv/bin/activate
```
### 2. Move to src folder
```bash
cd src
```

### 3. Run the program
```bash
python main.py <some .obj file>
```

### 4. Stop the program
```bash
^C
```
There is currently no way to stop the program cleanly. So keyboard interrupt it is.

### 5. Exit virtual environment
```bash
deactivate
```

## Models
Like said in description, the program can operate only with .obj format models. Other criteria is face amount and size of the model. I recommend to keep the face amount under 20k so the program can render it.

I added two models that I tested my self as default, [cat.obj](https://free3d.com/3d-model/cat-v1--522281.html) and [horse.obj](https://free3d.com/3d-model/american-paint-horse-nuetral-v1--575385.html). They can be found from assets folder by default. These models are published in [Free3D](https://free3d.com/) and they are made by printable_models.

If you wish to add your own 3D models, just make sure it is located in assets folder that can be found in project root.

## What to improve
1. Currently every rotation calculates face centers and their normal vectors for lighting all over. I do not think this is necessary at all, so in future I'll try calculating all normal vectors once, and apply the rotation matrix operation to them directly. This saves a lot of processing power

2. Models seem to have lot of white space (holes) in the middle of them. I think there is some bias in the way I round and typecast the coordinates to integers for the printing. I need to find out what the issue is there.

3. Overall features, like cancelling the animation, rotating the model initially and up- or downscaling the model would improve the experience. These features are coming sometime in the future.

4. Because this was sort of try it out kind of project the structure is a mess. In the future I should refactor some functions to only perform single task etc.

5. Cross platform support.


