# Research Project Installation Guide

## Author Information

- **Name:** Lucas Driessens
- **Institution:** HOWEST Kortrijk
- **Course:** Research Project
- **Date:** 2024-08-01

## Installation Steps

### Prerequisites

Ensure you have Python 3.11, pip, git, and Docker installed.

### Repository Setup

Clone and navigate to the repository:

```bash
git clone https://github.com/driessenslucas/researchproject.git
cd researchproject
```

### ESP32 Setup

#### Hardware Installation

- Refer to the hardware installation manual for ESP32 setup. [hardware_installation_guide.md](./hardware_installtion_guide.md).

#### Software Configuration

- Upload the code from the `esp32` folder to the ESP32.
- Modify the WiFi credentials in the code to your local network settings.

### Raspberry Pi (RPI) Setup

1. Install the Raspberry Pi OS on your Raspberry Pi 5. You can follow the official guide [here](https://www.raspberrypi.org/documentation/installation/installing-images/README.md).

#### Important Note

- Always execute `docker-compose down` after use to ensure proper virtual display startup.

#### Setup Instructions

- The code for the RPI is in the `main_web_app` folder.
- Run the following commands to start the Docker containers:

  ```bash
  cd main_web_app/
  docker-compose up -d
  ```

- Enable the script at startup to display the RPI IP address:

  ```bash
  sudo systemctl enable ./services/virtual_display.service
  sudo systemctl start virtual_display.service
  ```

### Camera Setup

- The camera script can be run either on a PC or another Raspberry Pi.
- Navigate to the `camera` folder and execute:

  ```bash
  docker-compose up -d
  ```

### Training

- Use the provided pre-trained model or train a new one using the "ResearchProject_v2.ipynb" Jupyter notebook in the main folder.
- The model is saved in the `models` folder and `main_web_app` folder. Restart the Docker container to launch the web app.

### Usage

- In the web app, enter the ESP's IP address and the IP of the device running the camera.
- You can opt for a virtual demonstration of the project without moving the car, primarily for demo purposes.

Demo of the web app using a virtual rc-car

https://github.com/driessenslucas/researchproject/assets/91117911/413fde11-6752-4a19-b113-028ed2151be2

