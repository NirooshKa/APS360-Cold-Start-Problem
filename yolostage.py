import subprocess
import os
import ntpath
import re
import sys
import matplotlib.pyplot as plt
from pathlib import Path

def DetectObjectsOnImage(targetPath):
    """Detect objects on a given input image
        - Pending exception handling
    """
    path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'darknet')
    process = subprocess.run("./darknet detector test cfg/coco.data yolov4.cfg yolov4.weights -ext_output " + f'"{targetPath}"', shell=True, capture_output=True, cwd=path)

    # Get the result
    filename = ntpath.basename(targetPath) # Not used
    reg = targetPath.replace('\\', '\\\\').replace('/', '\/').replace('.', '\\.') + r': Predicted .*'
    pattern = re.compile(reg, re.MULTILINE | re.DOTALL)
    result = pattern.search(process.stdout.decode('ascii').strip()).group()
    # print(result) # Debug use

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
    return coordinates

def SplitImageParts(sourceImage, coordinates):
    """Split image into subimages based on labeling
    """
    # Open output image
    img = plt.imread(sourceImage)
    # Split
    parts = []
    for c in coordinates:
        x, y, width, height = c['x'], c['y'], c['width'], c['height']
        crop = img[y:y+height, x:x+width, 0:3]
        parts.append(crop)
        # Temp: Save as output
        Path('work').mkdir(parents=False, exist_ok=True)
        plt.imsave(os.path.join('work', c['label'] + '.png'), crop)
    return parts

def ProcessImage(targetPath):
    """Process image at given path and return a bunch of matplotlib.pyplot.img's (numpy array)
        - Input `targetPath` must be absoluate path
    """
    # Get coordinates
    coordinates = DetectObjectsOnImage(targetPath)
    # Get splits
    return SplitImageParts(targetPath, coordinates)

if __name__ == '__main__':
    filename = 'data/dog.jpg'
    if len(sys.argv) == 2:
        filename = argv[1]
    # Run the detector
    targetPath = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'darknet', filename)
    print(f'Executing YOLO detector on: {targetPath}')
    coordinates = DetectObjectsOnImage(targetPath)
    print(f'Result:')
    for c in coordinates:
        print(f"{c['label']}: {c['confidence']:.2f}%\t(left_x: {c['x']}, top_y: {c['y']}, width: {c['width']}, height: {c['height']})")

    # Process coordinates
    results = SplitImageParts(targetPath, coordinates)
    # Crop a square region instead of a rectangular region...

    # Downstream process with handle results from here...
    # Downstram will use transformer to resize to 128x128