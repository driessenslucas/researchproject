#!/bin/bash

sleep 100

# Check if the Xvfb server is already running
if [ -e /tmp/.X99-lock ]; then
    echo "Display :99 is already in use. Trying to release it..."
    kill $(cat /tmp/.X99-lock)
    # Wait a bit to ensure the process has been killed
    sleep 2
fi

# Start Xvfb
Xvfb :99 -screen 0 1024x768x16 &
sleep 2 # Give some time for Xvfb to start
export DISPLAY=:99

# Allow local docker containers to connect to the X server
xhost +Local:docker