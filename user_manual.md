# Research Project Installation Guide

## Author Information

**Name:** Lucas Driessens  
**Institution:** Howest University of Applied Sciences  
**Course:** Research Project  
**Date:** 2024-30-01

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

- Refer to the hardware installation manual for ESP32 wire connections. [hardware installation guide](./hardware_installtion.md).

#### Software Configuration

- You will need to have this library installed if you want to use the OLED display [ESP32_SSD1306](https://github.com/lexus2k/ssd1306/tree/master)
- Upload the code from the [esp32](./esp32) folder to the ESP32.
- Modify the WiFi credentials in the code to your local network settings.

### Raspberry Pi (RPI) Setup

1. Install the Raspberry Pi OS on your Raspberry Pi 5. You can follow the official guide [here](https://www.raspberrypi.org/documentation/installation/installing-images/README.md).

#### Important Note

- Always execute `docker-compose down` after use to ensure proper virtual display startup.

#### Setup Instructions

- Make sure the i2c interface is enabled on the RPI. You can follow this guide [here](https://www.raspberrypi-spy.co.uk/2014/11/enabling-the-i2c-interface-on-the-raspberry-pi/).

- Install Docker on the RPI. You can follow this nice guide here [here](https://pimylifeup.com/raspberry-pi-docker/).

- The code for the RPI is in the [web app](./web_app/) folder from the main folder in the repository.

- Navigate to the [web app](./web_app/) folder.

  ```bash
  cd ./web_app/
  ```

- Run the following commands to start the Docker containers:

  ```bash
  cd ./web_app/
  docker-compose up -d
  ```

- Navigate to the [RPI_display](./RPI_display/) folder from the main folder in the repository.

  ```bash
  cd ./RPI_display/
  ```

- Enable the script at startup for the mini oled display to display the IP address of the RPI:

  ```bash
  sudo systemctl enable ./service/display_ip.service
  sudo systemctl start display_ip.service
  ```

### Training

- Use the provided pre-trained model or train a new one using the [train](./training/train.py) script.
- The [train](./training/train.py) script can be run on the RPI itself, it is not too resource intensive.
  - This will require you to have the packages in the [requirements.txt](./training/requirements.txt) file installed. Along with python@11.7 and pip.
- The script will ask you if you want to save the model. If you do, it will be saved in the [models](./models) folder.

### Usage

- In the web app, enter the ESP's IP address and select the model you want to use.
- Click on the `Start Maze` button to start the project.
- You can opt for a virtual demonstration of the project without moving the actual car.
