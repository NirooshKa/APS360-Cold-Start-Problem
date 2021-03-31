# Remove any existing server folder and archive
rm -fr Server
rm Server.zip
# Download new archive
# wget -O Server.zip https://github.com/NirooshKa/APS360-Cold-Start-Problem/releases/download/V0.2/Server.zip # Release with only YOLOStage
wget -O Server.zip https://github.com/NirooshKa/APS360-Cold-Start-Problem/releases/download/V0.3/Server.zip # Latest
unzip Server.zip -d Server
# Clean archive
rm Server.zip

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