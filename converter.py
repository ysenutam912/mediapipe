import sys
sys.path.append('mediapipe/framework/formats')
from detection_pb2 import Detection
from landmark_pb2 import LandmarkList
from rect_pb2 import NormalizedRect

#from mediapipe.framework.formats.detection_pb2 import Detection
#from mediapipe.framework.formats.landmark_pb2 import LandmarkList
#from mediapipe.framework.formats.rect_pb2 import NormalizedRect
from google.protobuf.json_format import MessageToJson

import glob
import re
from pprint import pprint
import json
# import sys

if len(sys.argv) > 1:
    targetDir = sys.argv[1]
else:
    targetDir = "./result/4hands_output.mov/"

outputFiles = glob.glob(targetDir + "/" + "*.txt")


detectionFiles = [(re.findall(
    r"FRAMENUM=(\d+)_detection_j=(\d+).txt", outputFile), outputFile) for outputFile in outputFiles]
detectionFilesFiltered = [
    (detectionFile[0], detectionFile[1].replace("\\", "/")) for detectionFile in detectionFiles if detectionFile[0]]

landmarkFiles = [(re.findall(
    r"FRAMENUM=(\d+)_landmark_j=(\d+).txt", outputFile), outputFile) for outputFile in outputFiles]
landmarkFilesFiltered = [
    (landmarkFile[0], landmarkFile[1].replace("\\", "/")) for landmarkFile in landmarkFiles if landmarkFile[0]]

handRectFiles = [(re.findall(
    r"FRAMENUM=(\d+)_handRect_j=(\d+).txt", outputFile), outputFile) for outputFile in outputFiles]
handRectFilesFiltered = [
    (handRectFile[0], handRectFile[1].replace("\\", "/")) for handRectFile in handRectFiles if handRectFile[0]]

palmRectFiles = [(re.findall(
    r"FRAMENUM=(\d+)_palmRect_j=(\d+).txt", outputFile), outputFile) for outputFile in outputFiles]
palmRectFilesFiltered = [
    (palmRectFile[0], palmRectFile[1].replace("\\", "/")) for palmRectFile in palmRectFiles if palmRectFile[0]]

handRectFromLandmarksFiles = [(re.findall(
    r"FRAMENUM=(\d+)_handRectFromLandmarks_j=(\d+).txt", outputFile), outputFile) for outputFile in outputFiles]
handRectFromLandmarksFilesFiltered = [
    (handRectFromLandmarksFile[0], handRectFromLandmarksFile[1].replace("\\", "/")) for handRectFromLandmarksFile in handRectFromLandmarksFiles if handRectFromLandmarksFile[0]]


for detectionFile in detectionFilesFiltered:
    with open(detectionFile[1], "rb") as f:
        content = f.read()

    detection = Detection()
    detection.ParseFromString(content)
    jsonObj = MessageToJson(detection)

    detectionFileOutput = detectionFile[1].replace("txt", "json")
    with open(detectionFileOutput, "w") as f:
        f.write(jsonObj) 


for landmarkFile in landmarkFilesFiltered:
    with open(landmarkFile[1], "rb") as f:
        content = f.read()

    landmark = LandmarkList()
    landmark.ParseFromString(content)
    jsonObj = MessageToJson(landmark)

    landmarkFileOutput = landmarkFile[1].replace("txt", "json")
    with open(landmarkFileOutput, "w") as f:
        f.write(jsonObj) 


for handRectFile in handRectFilesFiltered:
    with open(handRectFile[1], "rb") as f:
        content = f.read()

    handRect = NormalizedRect()
    handRect.ParseFromString(content)
    jsonObj = MessageToJson(handRect)

    handRectFileOutput = handRectFile[1].replace("txt", "json")
    with open(handRectFileOutput, "w") as f:
        f.write(jsonObj) 


for palmRectFile in palmRectFilesFiltered:
    with open(palmRectFile[1], "rb") as f:
        content = f.read()

    palmRect = NormalizedRect()
    palmRect.ParseFromString(content)
    jsonObj = MessageToJson(palmRect)

    palmRectFileOutput = palmRectFile[1].replace("txt", "json")
    with open(palmRectFileOutput, "w") as f:
        f.write(jsonObj) 

for handRectFromLandmarksFile in handRectFromLandmarksFilesFiltered:
    with open(handRectFromLandmarksFile[1], "rb") as f:
        content = f.read()

    handRectFromLandmarks = NormalizedRect()
    handRectFromLandmarks.ParseFromString(content)
    jsonObj = MessageToJson(handRectFromLandmarks)

    handRectFromLandmarksFileOutput = handRectFromLandmarksFile[1].replace("txt", "json")
    with open(handRectFromLandmarksFileOutput, "w") as f:
        f.write(jsonObj)
