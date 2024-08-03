# MetaBCI

## Welcome! 
MetaBCI is an open-source platform for non-invasive brain computer interface. The project of MetaBCI is led by Prof. Minpeng Xu from Tianjin University, China. MetaBCI has 3 main parts:
* brainda: for importing dataset, pre-processing EEG data and implementing EEG decoding algorithms.
* brainflow: a high speed EEG online data processing framework.
* brainstim: a simple and efficient BCI experiment paradigms design module. 

This version is primarily built upon the existing MetaBCI functions, expanding the integration of multi-brand BCI device drivers to achieve synchronous collection of heterogeneous devices and multiple individuals trigger management. Through the collaborative "hyperscanning" paradigm, it integrates multi-user EEG information to study the collaborative effect on decoding accuracy and speed of BCI, providing a novel approach for practical application. Additionally, the project establishes a platform for acquiring and decoding multi-human brain electrical data while offering interfaces for various products from different brands, thereby enhancing convenience in device usage.

## Content

- [MetaBCI](#metabci)
  - [Welcome!](#welcome)
  - [What are we doing?](#what-are-we-doing)
    - [The problem](#the-problem)
    - [The solution](#the-solution)
  - [Who are we?](#who-are-we)
  - [License](#license)
  - [Contact](#contact)
  - [Acknowledgements](#acknowledgements)

## What are we doing?

### The problem

* The current BCI system predominantly relies on the EEG signal of an individual user, leading to an efficiency bottleneck
* Different brands of EEG devices employ varying data collection and utilization methods, which causes inconvenience to use

### The solution

This version will:

* Establishment of a platform to support the simultaneous acquisition of EEG from multiple brands of devices and multiple individuals
* Provides multiple individuals SSVEP collaborative decoding examples
* Ensure simultaneous acquisition of EEG from multiple devices and synchronization of trigger across multiple devices based on Brainflow

## Who are we?

We are undergraduates from mindlab, szu.

## License

Distributed under the GNU General Public License v2.0 License. See `LICENSE` for more information.

## Contact

Email: TBC_TJU_2022@163.com

## Acknowledgements
感谢黄淦老师、天津大学梅老师、念通智能工程师等的鼎力相助。
