# hw3_1

mbed run a RPC loop with two custom functions 
  (1) gesture UI, and (2) tilt angle detection
PC/Python use RPC over serial to send a command to call gesture UI mode on mbed
In the thread function, user will use gesture to select from a few threshold angles.
After the selection is confirmed with a user button, the selected threshold angle is published through WiFi/MQTT to a broker (run on PC).
After the PC/Python get the published confirmation from the broker, it sends a command to mbed to stop the guest UI mode. 
Therefore, the mbed is back to RPC loop. Also PC/Python will show the selection on screen.
PC/Python use RPC over serial to send a command to call tilt angle detection mode on mbed
The tilt angle function will start a thread function.
If the tilt angle is over the selected threshold angle, mbed will publish the event and angle through WiFi/MQTT to a broker.
After the PC/Python get a preset number 10 tilt events, it sends a command to mbed to stop the tilt detection mode. 
Therefore, the mbed is back to RPC loop. Also PC/Python will show all tilt events on screen.
