# A development note of *Brainflow*

## 1. Updating note

Now, *brainflow* is able to support more kinds of devices by:

* Native support Neuroscan and Neurcale by inheriting and re-written the BaseAmplifier class.
* Indirect support devices, such as [EGI](https://www.egi.com), [g.tec](https://www.gtec.at/) and [BioSemi](https://www.biosemi.com), by implementing a communication protocol between MetaBCI and [Lab streaming layer](https://github.com/sccn/labstreaminglayer). Due to the LSL provides many specific apps for retrieving data from different devices in a unified format, the MetaBCI could get and process online data from these devices by using LSL.
* For meeting the requirements of recording event triggers without hardware like Serial and LPT. We provide a feasible way to writing events triggers (also known as markers). Users can use it to write markers just like writing hardware triggers.

## 2. How to use

* You need to check [this website](https://labstreaminglayer.readthedocs.io/info/supported_devices.html) and download the LSL app for your device.
* Write some code to instantiate the LSLapp class.
* The remaining steps are similar with demo scripts [here]()

## 3. Suggestion and limitation

* If it is possible, **TRY TO USE HARDWARE EVENT TRIGGER**. As far as I know, the hardware event triggers enable more precise synchronization of events with the device data streams, in most of the situation. So if you had a trigger box or any other similar devices, try to use it without applying the software event trigger.
* If you owned a physical device, and also able to acquire the communication protocol code or instructions for getting data from the devices. We strongly suggest that you can try to inherit and re-write the BaseAmplifier class for help MetaBCI to native more device.
* Our lab (TUNERL) owned an EGI device, the metabci team planned to support EGI first.
* Considering the differences among different devices for transfering the event trigger.
    **YOU MUST BE VERY CAREFUL** to determine wethher the data stream reading
    from the LSL apps contains a event channel. For example, the neuroscan
    synamp II will append a extra event channel to the raw data channel.
    Because we do not have chance to test each device that LSL supported, so
    please modify this class before using with your own condition.