# Research Project

## Author Information

**Name:** Lucas Driessens  
**Institution:** HOWEST Kortrijk  
**Course:** Research Project  
**Date:** 2024-08-01

## Description

### Main Research Question

> Is it possible to transfer a trained RL-agent from a simulation to the real world? (case: maze)

### Sub Research Questions

1. Which virtual environments exist to train a virtual RF-car?

2. Which reinforcement learning techniques can I best use in this application?

3. Can the simulation be transferred to the real world? Difference between how the car moves in the simulation and in the real world.

4. Does the simulation have any useful contributions? In terms of training time or performance?

5. How can I transfer my trained model to my physical RC car? (sim2real) How do I adjust the agent, environment and model to work in the real world?

6. How can Real-time learning be implemented?

7. Extra: How can I make the car drive through an unknown maze?

## Installation

- prerequisites: python 3.11, pip, git, docker...

```bash

git clone https://github.com/driessenslucas/researchproject.git && cd researchproject

```

### ESP32 setup

- after looking at the hardware installation manual, you can continue by uploading the code to the esp found in the "esp32" folder of the project

- you will need to change the wifi credentials in the code to match your own local network

### RPI setup

**please note to always run docker-compose down after running. This is to make sure the virtual display will be albe to start up correctly**

- the code ran on the RPI is found in the "main web app" folder..

```bash

cd main_web_app/ && docker-compose up -d

```

- you should also enable the script to run on startup, so you can see the ip address the rpi is running on

```bash

sudo systemctl enable ./services/virtual_display.service && sudo systemctl start virtual_display.service

```

### camera setup for the camera

- the camera script can either be ran on the computer itself or anther raspberry pi ( like I did )

- go to the "camera" folder and run the command

```bash

docker-compose up -d

```

## Training

- you can use my pre-trained model, or train your own model. The jupyter notebook can be found in the main folder of the project under the name "ResearchProject_v2.ipynb".

After running the whole script the model will be saved in the "models" folder and also in the main_web_app folder. You can then run the docker container again to start the web app.

## Usage

- on the webapp you will need to enter the IP address displayed on the little screen of the esp. and the IP address of the device running the camera docker container

Next you can choose to either run the virtualization of the project ( without actually moving the car, this is used more as a demo and to see how the car would move)

![rc_control_demo](./video/maze_web_app.mp4)
