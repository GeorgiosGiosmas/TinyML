# Plants Village Classification problem with Edge Impulse and Pytorch and Inference of STM32F407G DISC1

## Introduction
In this project we will examine a Classification problem. We will take a specific dataset, train some custom Neural Networks on this dataset and then try to run some of our trained models on our STM32F407G DISC1 board. Running inference of Neural Networks on Microcontrollers raises some challenges regarding the size of these networks, as some of them isn't possible to be run on such boards due to Flash Memory and RAM requirements. Our STM32F407G DISC1 board disposes 1 MB of Flash and 192 KBs of RAM, so out of all the trained models we will acquire only those meeting the size requirements are going to be run on the Microcontroller Board.

