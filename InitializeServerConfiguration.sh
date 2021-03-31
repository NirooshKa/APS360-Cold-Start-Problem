printf "PythonPath: $(which python3)\n" > ServerConfigurations.yaml
printf "PythonCommands: [\"$(pwd)/yolostage.py\", \"$(pwd)/work/work.png\"]" >> ServerConfigurations.yaml
apt-get update && apt-get install -y libgdiplus
pip3 install torch==1.8.1+cpu torchvision==0.9.1+cpu torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html