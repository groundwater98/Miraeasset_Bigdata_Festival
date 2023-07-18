import sys
import os

script_dir = os.path.dirname(os.path.abspath(__file__))

relative_path = "../miraeasset-festa/DataEngineering/srcs"

target_dir = os.path.join(script_dir, relative_path)
target_dir = os.path.normpath(target_dir)

sys.path.append(target_dir)