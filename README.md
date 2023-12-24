# Vietnamese Text Recognition using Paddle OCR

This open-source repository is dedicated to the recognition of Vietnamese words using the PaddleOCR framework. Developed by our AI team, the project focuses on creating a specialized text recognition model tailored for the Vietnamese language, with a particular emphasis on outdoor images.

## The problem

Recognizing Vietnamese words in outdoor photos and daily activities is a necessary issue in the boom period of Artificial Intelligence. The main goal of the contest is to detect and recognize text in images, especially focusing on identifying text in scene text collected from many different camera sources. all over Vietnam.

This problem is important for many modern artificial intelligence systems, such as robots and autonomous vehicles. These systems require the ability to understand the surrounding landscape, especially the ability to recognize words in the landscape, which contain a lot of important information. The success of the solution can support the development of smart applications in tourism, museums, automated vehicles and autonomous robots.

To increase the practical applicability of the model, the requirement is not only accuracy but also fast processing time.

## Trained models

Due to Github repository limitations, we stored the model separately in a dedicated Google Drive repository. To be able to use the source code, users need to download these folders to their computer, unzip them and put the folders into the main directory of the Github repo so that the source code can identify the model. The model for this project is in 2 folders:

- [SAST](https://drive.google.com/drive/folders/1NNUDgZvCnJrjoonCadp7nRYHxkWgNJyt?usp=sharing) : folder stores the parameters of the detection model, using the SAST algorithm.
- [SRN](https://drive.google.com/drive/folders/1JpJ1o2cyYunNDhuW3sbqx17EVDAAbwES?usp=sharing) : folder stores the parameters of the recognition model, using the SRN algorithm.
