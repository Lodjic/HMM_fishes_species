# Hidden Markov Models assignment

# Introduction

This repository concerns an assignment of a KTH course called "Artificial Intelligence DD2380". The objective is to implement Hidden Markov Models (HMM) in order to guess a maximum of species of each of the 70 fishes present in an environnement. There are 7 different sepcies we do not know the number of fishes per species.

# Objective
The objective is to implement Hidden Markov Models (HMM) in order to guess a maximum of species of each of the 70 fishes present in an environnement.

The solution needed to be efficient, with a time limit per cycle of 5 seconds. 
The graphical interface spends 0.5 seconds per cycle and waits for player's response if the solution takes longer time.
In each cycle, all the fishes do one move and the game allows a maximum of 180 cycles.

# Solution description

The solution is provided in the file `player.py`.
To be able to guess the species of the each fish, the algorithm constructs 7 HMMs for the 7 different species. First, it waits for a certain number of cycles to observe and register the movement of all the fishes. Then, the algorithm tries to guess the species of one random fish per cycle and adjust the matrixes of the corresponding HMM if the species was wrongly guessed.


# Given code
The skeleton provided includes an implementation of the KTH Fishing Derby. 


# Manual Installation and run

The code runs in Python 3.7 AMD64 or Python 3.6 AMD64.

## Installation

You should start with a clean virtual environment and install the requirements for the code to run. You may create a Python 3.6 or Python 3.7 environment and install the required packages (`requirements.txt` for UNIX or `requirements_win.txt` for Windows).
 
I used a Mac to do it. So, on Mac OS X:

1. Install **python 3.7** or **python 3.6**

   https://www.python.org/downloads/mac-osx/

2. Install **virtualenv** and if you want **virtualenvwrapper**

   * Install them with pip3.

   ```
   $ sudo pip3 install virtualenv
   $ sudo pip3 install virtualenvwrapper
   ```

3. Add your virtualenvwrapper.sh to you bash_profile and/or create your virtual environment 'fishingderby' and install the requirements :

   ```
   (fishingderby) $ pip3 install -r requirements.txt
   ```

# Graphical Interface

To run the GUI, in the skeleton's directory, run in the terminal:

```
(fishingderby) $ python3 main.py < sequences.json
```

# Note

Pure python was used in this projects because the length of the list are quite small and the software on which this code was run for evaluation did not support numpy. 
