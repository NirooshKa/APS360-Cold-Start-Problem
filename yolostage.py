import subprocess
import os
import ntpath
import re

# Run the detector
targetPath = "data/dog.jpg"
path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'darknet')
process = subprocess.run("./darknet detector test cfg/coco.data yolov4.cfg yolov4.weights -ext_output " + targetPath, shell=True, capture_output=True, cwd=path)

# Get the result
filename = ntpath.basename(targetPath)
reg = targetPath.replace('\\', '\\\\').replace('/', '\/').replace('.', '\\.') + r': Predicted .*'
pattern = re.compile(reg, re.MULTILINE | re.DOTALL)
result = pattern.search(process.stdout.decode('ascii').strip()).group()
print(result)

# Parse result
coordinates = []
for line in result.splitlines():
    # Skip first line of output
    if line.startswith(targetPath):
        continue
    # Get coordinates of matched results
    else:
        match = re.search(r'(?P<label>[^:]+): (?P<confidence>\d+)%\s*\((?P<coord>.*?)\)', line)
        coord_string = re.search(r'left_x:\s+(?P<x>\d+)\s+top_y:\s+(?P<y>\d+)\s+width:\s+(?P<w>\d+)\s+height:\s+(?P<h>\d+)\s*', match.group('coord'))
        x,y,w,h = int(coord_string.group('x')), int(coord_string.group('y')), int(coord_string.group('w')), int(coord_string.group('h'))
        coordinates.append({"label": match.group('label'), "confidence": float(match.group('confidence')), "x": x, "y": y, "width": w, "height": h})

print(coordinates)