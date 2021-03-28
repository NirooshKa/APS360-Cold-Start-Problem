printf "PythonPath: $(which python3)\n" > ServerConfigurations.yaml
printf "PythonCommands: [$(pwd)/yolostage.py]" >> ServerConfigurations.yaml