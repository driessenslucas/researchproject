#!/bin/bash

# Check if the Xvfb server is already running
if [ -e /tmp/.X100-lock ]; then
    echo "Display :100 is already in use."

    # Check if the process listed in the lock file is actually running
    if ps -p $(cat /tmp/.X100-lock) > /dev/null 2>&1; then
       echo "Xvfb is already running. Proceeding to set DISPLAY."
    else
       echo "Lock file found but Xvfb is not running. Starting Xvfb."
       # Start Xvfb
       Xvfb :100 -screen 0 1024x768x16 &
       sleep 2 # Give some time for Xvfb to start
    fi
else
    # Start Xvfb since it's not running
    echo "Starting Xvfb."
    Xvfb :100 -screen 0 1024x768x16 &
    sleep 2 # Give some time for Xvfb to start
fi

# Set DISPLAY
export DISPLAY=:100

# Allow local docker containers to connect to the X server
# This should be done only if Xvfb is running
if xhost > /dev/null 2>&1; then
    xhost +Local:docker
else
    echo "Xvfb is not running. Cannot set xhost."
fi

# Add a success message
echo "Virtual display setup completed successfully. Display: ${DISPLAY}"
