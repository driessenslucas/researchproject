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

- Refer to the hardware installation manual for ESP32 setup. [hardware installation guide](./hardware_installtion.md).

#### Software Configuration

- You will need to have this library installed if you want to use the OLED display [ESP32_SSD1306](https://github.com/lexus2k/ssd1306/tree/master)
- Upload the code from the [esp32](./esp32) folder to the ESP32.
- Modify the WiFi credentials in the code to your local network settings.

### Raspberry Pi (RPI) Setup

1. Install the Raspberry Pi OS on your Raspberry Pi 5. You can follow the official guide [here](https://www.raspberrypi.org/documentation/installation/installing-images/README.md).

#### Important Note

- Always execute `docker-compose down` after use to ensure proper virtual display startup.

#### Setup Instructions

- The code for the RPI is in the [web app](./web_app/) folder.
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
- Navigate to the [camera stream](./camera_stream) folder and execute:

  ```bash
  docker-compose up -d
  ```

### Training

- Use the provided pre-trained model or train a new one using the [train](./training/train.py) script.
- The [train](./training/train.py) script can be run either on a PC or the RPI itself.
- The script will ask you if you want to save the model. If you do, it will be saved in the [models](./models) folder.

### Usage

- In the web app, enter the ESP's IP address and select the model you want to use.
- Click on the `Start Maze` button to start the project.
- You can opt for a virtual demonstration of the project without moving the car, primarily for demo purposes.
