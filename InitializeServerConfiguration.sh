printf "PythonPath: $(which python3)\n" > ServerConfigurations.yaml
printf "PythonCommands: [\"$(pwd)/yolostage.py\", \"$(pwd)/work/work.png\"]" >> ServerConfigurations.yaml
apt-get update && apt-get install -y libgdiplus
