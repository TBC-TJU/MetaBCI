# MetaBCI

## Welcome! 
This project, based on the open-source MetaBCI platform, has established an immersive VR rehabilitation training system that achieves full-chain domestic technological localization. By leveraging a domestic high-performance 3D engine, it constructs high-realism virtual rehabilitation scenarios to present rehabilitation tasks in real time and provide synchronized feedback on motor imagery decoding results, thereby enhancing patients' focus during training. By integrating the BrainCo EEG acquisition system and the built-in FBCSP algorithm in the MetaBCI platform, it realizes real-time transmission and recognition of EEG signals, with an instruction recognition accuracy rate reaching 82.5%. The system drives electrical stimulation devices based on decoded instructions, triggering graded contraction of specific hand muscles in patients (with adjustable intensity), effectively enhancing both training focus and rehabilitation outcomes. From 3D scene rendering, EEG signal acquisition and decoding to electrical stimulation execution, the entire technical chain employs domestically developed technology and equipment, truly establishing a fully domestic immersive VR rehabilitation system that integrates "sensing-computing-intervention."

## Content

- [MetaBCI](#metabci)
  - [Welcome!](#welcome)
  - [What are we doing?](#what-are-we-doing)
  - [Section on Updates and Fixes](#section-on-updates-and-fixes)
  - [Usage Instructions](#usage-instructions)
  - [Installation](#installation)
  - [Who are we?](#who-are-we)
  - [Contact](#contact)
  - [Acknowledgements](#acknowledgements)

## What are we doing?

* Relying on the open-source platform architecture of MetaBCI, this project efficiently integrates the VR stimulation feedback interface, BrainCo EEG acquisition system communication interface, real-time Motor Imagery (MI) decoding algorithm, and self-developed electrical stimulation device control module to implement a rehabilitation training system for patients with motor disorders that combines VR and electrical stimulation technologies.
  - Stimulation Presentation​​: Achieves real-time communication between the host system and VR devices. The VR headset displays the rehabilitation task interface and provides real-time feedback on MI decoding results to enhance the immersive training experience.
  - Data Acquisition​​: Improves the BrainCo data parsing module on the BrainFlow sub-platform to enable real-time acquisition and transmission of EEG signals to the processing terminal.
  - ​​Signal Processing​​: Invokes the FBCSP algorithm from the BrainDA sub-platform to perform real-time feature extraction and pattern recognition on EEG signals, decoding users' motor intentions into control commands.
  - Peripheral Device Control​​: Newly adds an electrical stimulation device control module on the BrainFlow sub-platform to control the electrical stimulation device based on the decoded commands, triggering contraction of specific hand muscles in patients and achieving the goal of closed-loop rehabilitation training.

## Section on Updates and Fixes

* Updates
   - Add Electrical Stimulation Control Module    Brainflow	  metabci\brainflow\ ElectroStimulator.py    1.ElectroStimulator()
       - demo:  demos\brainflow_demos\FES.py
   - Add Stimulation Tag Transmission Function    Brainflow	  metabci\brainflow\amplifiers.py    1.BaseAmplifier()  2.Marker()
       - demo:  demos\brainflow_demos\Online_mi_nc.py

* Fixes
   - Optimize Neuracle Amplifier Data Stream Module    Brainflow    metabci\brainflow\amplifiers.py    1.Neuracle()
     demo:  demos\brainflow_demos\Online_mi_nc.py
     
##  Usage Instructions

  - On Stimulation Computer A, first run the Blank_stim.py file, then open the VR stimulation host program TunerlRehabilitation.exe;
  - On EEG Reception and Recognition Computer B, open the BrainCo acquisition program, click "Data Forwarding," and run the MIprocess_Online.py file;
  - Set the rehabilitation mode on Computer A: sequentially click Mode Selection, Start Rehabilitation, and Specified Task Rehabilitation. Set both Left Turn and Right Turn to 10 times, then click Start Rehabilitation.




## Installation

1. Clone the repo
   ```sh
   git clone https://github.com/TBC-TJU/MetaBCI.git
   ```
2. Change to the project directory
   ```sh
   cd MetaBCI
   ```
3. Install all requirements
   ```sh
   pip install -r requirements.txt 
   ```
4. Install brainda package with the editable mode
   ```sh
   pip install -e .
   ```
## Who are we?

The MetaBCI project is carried out by researchers from 
- Academy of Medical Engineering and Translational Medicine, Tianjin University, China
- Tianjin Brain Center, China


## Contact

Email: 1364747481@qq.com

## Acknowledgements
- [MNE](https://github.com/mne-tools/mne-python)
- [MOABB](https://github.com/NeuroTechX/moabb)
- [pyRiemann](https://github.com/alexandrebarachant/pyRiemann)
- [TRCA/eTRCA](https://github.com/mnakanishi/TRCA-SSVEP)
- [EEGNet](https://github.com/vlawhern/arl-eegmodels)
- [RPA](https://github.com/plcrodrigues/RPA)
- [MEKT](https://github.com/chamwen/MEKT)
