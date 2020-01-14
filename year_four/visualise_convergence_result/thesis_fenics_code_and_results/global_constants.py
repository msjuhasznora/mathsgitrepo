from dolfin import *
import datetime
import argparse

global epsilon_lower_limit
epsilon_lower_limit = 1.0e-07 #up 1.0e-07 c 5e-04

global verbose
verbose = True

global timestamp
timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

global default_mesh
default_mesh = UnitSquareMesh(30, 30)

global resultsfolder
parser = argparse.ArgumentParser()
parser.add_argument("-r", "--resultfolder", default="current_results_H_V", help="default: results, custom: name of results folder")
xargs = parser.parse_args(None)
resultsfolder = str(timestamp) + xargs.resultfolder + "/"
