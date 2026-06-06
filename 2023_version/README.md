# GA_Drawing_Example

<p align="center">
 <img src="2023_version/imgs/ga-example.PNG" width="400" height="400" >
</p>



## Table of contents
* [Project Description](#Project-Description)
* [Requirements](#Requirements)
* [How to Use](#How-to-Use)
* [Results](#Results)
* [Future Work](#Future-Work)
* [References](#References)


## Project Description
This project is modified from the code presented in the post "Painting the Starry Night with Genetic Algorithms" by 
Samuel Hinton (2021-03-19) at https://cosmiccoding.com.au/tutorials/genetic_part_one . It now also includes part two, 
"Genetic Algorithms 2: Girl with a Pearl Earring" from https://cosmiccoding.com.au/tutorials/genetic_part_two

This version is designed to work with wxPython instead of pygame to integrate into a currently existing machine learning
GUI example. Other major changes that have been made include implementing a more rigid class-based structure to increase
reusability of individual classes, and plans for side-by-side comparisons of the effect of different tuning parameters 
on evolving images.


## Requirements
Library requirements for locally run Python code are included in requirements.txt and can be 
installed using 'pip install -r requirements.txt'

* colour==0.1.5
* contourpy==1.0.5
* cycler==0.11.0
* fonttools==4.37.4
* kiwisolver==1.4.4
* matplotlib==3.6.1
* numpy==1.23.4
* packaging==21.3
* Pillow==9.2.0
* pyparsing==3.0.9
* python-dateutil==2.8.2
* six==1.16.0
* wxPython==4.2.0

## How to Use
The program can be run from main.py, either in an IDE or with 'python main.py'


TODO: step-by-step implementation examples and tuning

## Results

TODO: GIFs/Images with params for examples of images drawn with the GA 

### Future Work
This example was designed to use a wxPython GUI instead of the popular pygame as a proof of concept
for expanding into a currently existing machine learning demo GUI. However, this project is designed to 
stand alone and has the following future work planned:
* options for drawing multiple images in side-by-side comparison
* documentation update with expanded references for further reading
* code clean up for multi-organism (population2)  + increased control from user settings
* cell inspiration from: https://blog.4dcu.be/programming/2020/02/10/Genetic-Art-Algorithm-2.html
* interative UI for selecting start/pause/stop/save and single/multi organism options rather than using main.py for all configurations

### Added features since initial commit
* options for multiple organisms
* editing for consistency in vocab of variables and function calls

## References

Original genetic algorithm drawing tutorial
* https://cosmiccoding.com.au/tutorials/genetic_part_one
* https://cosmiccoding.com.au/tutorials/genetic_part_two

Images:
* Starry Night, Vincent van Gogh
* Girl with a Pearl Earring, from Susan Herbert, Cats Galore: a Compendium of Cultured Cats, featured in https://www.huffpost.com/entry/cats-are-taking-over-famous-western-artworks_n_55e75737e4b0aec9f355c018, 



