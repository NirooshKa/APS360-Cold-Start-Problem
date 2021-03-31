# Remove any existing server folder and archive
rm -fr Server
rm Server.zip
# Download new archive
# wget -O Server.zip https://github.com/NirooshKa/APS360-Cold-Start-Problem/releases/download/V0.2/Server.zip # Release with only YOLOStage
wget -O Server.zip https://github.com/NirooshKa/APS360-Cold-Start-Problem/releases/download/V0.3/Server.zip # Latest
unzip Server.zip -d Server
# Clean archive
rm Server.zip