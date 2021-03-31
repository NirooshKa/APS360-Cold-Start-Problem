# Remove any existing weight files
rm cats.pt
rm cats_and_dogs.pt
rm horses.pt
rm flowers.pt
# Download new weight files
# wget -O cats.pt https://github.com/NirooshKa/APS360-Cold-Start-Problem/releases/download/V0.3/cats.pt
wget -O cats_and_dogs.pt https://github.com/NirooshKa/APS360-Cold-Start-Problem/releases/download/V0.3/cats_and_dogs.pt
wget -O horses.pt https://github.com/NirooshKa/APS360-Cold-Start-Problem/releases/download/V0.3/horses.pt
wget -O flowers.pt https://github.com/NirooshKa/APS360-Cold-Start-Problem/releases/download/V0.3/flowers.pt