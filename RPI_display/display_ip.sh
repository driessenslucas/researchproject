#!/bin/bash

# Clear the display
./ssd1306_linux/ssd1306_bin -n 1 -c

# Getting the IP address
IP_ADDRESS=$(hostname -I | cut -d' ' -f1)

# Check if the IP address was successfully obtained
if [ -z "$IP_ADDRESS" ]; then
    echo "Error: Could not obtain IP address."
    exit 1
fi

# Display the label for the IP address
./ssd1306_linux/ssd1306_bin -n 1 -x 1 -y 1 -l "IP address RPI:"

# Position and display the actual IP address on the next line
./ssd1306_linux/ssd1306_bin -n 1 -x 1 -y 2 -l "$IP_ADDRESS"

# Check if the display command was successful
if [ $? -ne 0 ]; then
    echo "Error: Could not display IP address on the SSD1306."
    exit 2
fi

echo "IP address displayed successfully."
