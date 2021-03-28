printf "PythonPath: $(which python3)\n" > ServerConfigurations.yaml
printf "PythonCommands: [\"$(pwd)/yolostage.py\", \"work/work.png\"]" >> ServerConfigurations.yaml
