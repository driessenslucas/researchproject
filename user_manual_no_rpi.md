# Research Project Installation Guide

## Author Information

**Name:** Lucas Driessens  
**Institution:** Howest University of Applied Sciences  
**Course:** Research Project  
**Date:** 2024-30-01

## Installation Steps

### Prerequisites

Ensure you have git and Docker installed. (optionally you can install Python 3.11 and pip along with the packages in the [requirements.txt](./training/requirements.txt) file if you want to be able to train your own model)

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

### Web App Setup

#### Important Note

- Always execute `docker-compose down` after each use to ensure proper virtual display startup.

#### Setup Instructions

- The code for the web app is in the [web app](./web_app/) folder.

- Navigate to the [web app](./web_app/) folder.

  ```bash
  cd ./web_app/
  ```

- Run the following commands to start the Docker containers:

  ```bash
  cd ./web_app/
  docker-compose up -d
  ```

### Usage

1. In the web app, enter the ESP's IP address and select the model you want to use.
2. You can opt for a virtual demonstration of the project without moving the actual car.
3. Click on the `Start Maze` button to start.

A demo can be found: <https://github.com/driessenslucas/researchproject/assets/91117911/b440b295-6430-4401-845a-a94186a9345f>

### EXTRA: Training

- Use the provided pre-trained model or train a new one using the [train](./training/train.py) script.
- The [train](./training/train.py) script can be run on the RPI itself, it is not too resource intensive.
- The script will ask you if you want to save the model. If you do, it will be saved in the [models](./web_app/models) folder in de `web_app` directory.
