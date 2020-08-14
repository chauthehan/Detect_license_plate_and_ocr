import os

BASE_PATH = "dlib_front_and_rear_vehicles_v1"

TRAIN_XML = os.path.sep.join([BASE_PATH, "training.xml"])
TEST_XML = os.path.sep.join([BASE_PATH, 'testing.xml'])

TRAIN_RECORD = os.path.sep.join([BASE_PATH, "records/training.record"])
TEST_RECORD = os.path.sep.join([BASE_PATH, "records/testing.record"])
CLASSES_FILE = os.path.sep.join([BASE_PATH, "records/classes.pbtxt"])

CLASSES = {'rear': 1, 'front': 2}