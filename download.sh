#!/bin/bash
# Download and unzip datasets
wget https://github.com/NirooshKa/APS360-Cold-Start-Problem/releases/download/V0.1/APS360ProjectDataset_Cats.zip
wget https://github.com/NirooshKa/APS360-Cold-Start-Problem/releases/download/V0.1/APS360ProjectDataset_CatsAndDogs.zip
wget https://github.com/NirooshKa/APS360-Cold-Start-Problem/releases/download/V0.1/APS360ProjectDataset_Flowers.zip
wget https://github.com/NirooshKa/APS360-Cold-Start-Problem/releases/download/V0.1/APS360ProjectDataset_Horses.zip
unzip APS360ProjectDataset_Cats.zip
unzip APS360ProjectDataset_CatsAndDogs.zip
unzip APS360ProjectDataset_Flowers.zip
unzip APS360ProjectDataset_Horses.zip
rm APS360ProjectDataset_Cats.zip
rm APS360ProjectDataset_CatsAndDogs.zip
rm APS360ProjectDataset_Flowers.zip
rm APS360ProjectDataset_Horses.zip