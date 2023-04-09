import argparse
import sys
import numpy as np

import os, time
import math
import re
import tempfile
from tempfile import mkdtemp
from subprocess import Popen, check_output,STDOUT,run
import pandas as pd
import gzip
from os.path import splitext, basename, exists, abspath, isfile, getsize

def main(argv=sys.argv):
    parser = argparse.ArgumentParser(description='InDEP')
    parser.add_argument("-i", dest='input', default="./mc3.v0.2.8.PUBLIC.maf", help="fea")
    parser.add_argument("-o", dest='out', default="tcga_18.maf", help="fea")
    parser.add_argument("-m", dest='mode', default="train", help="mode") #train,score,eval
    args = parser.parse_args()

    if args.mode == 'fea':
            cmd = 'python run.py -t PANCAN -m fea -l InDEP '
            print(cmd)
            o = check_output(cmd, shell=True, universal_newlines=True)
            print(o)
    elif args.mode == 'train':
            cmd = 'python run.py -t PANCAN -m train -l InDEP'
            print(cmd)
            o = check_output(cmd, shell=True, universal_newlines=True)
            print(o)
    elif args.mode == 'score':
            cmd = 'python run.py -t PANCAN -m score -l InDEP'
            print(cmd)
            o = check_output(cmd, shell=True, universal_newlines=True)
            print(o)
    elif args.mode == 'eval':
            cmd = 'python run.py -t PANCAN -m eval -l InDEP'
            print(cmd)
            o = check_output(cmd, shell=True,universal_newlines=True)
            print(o)
    if args.mode == 'interpretation':
            cmd_te = 'python run.py -t PANCAN -m interpretation  -l InDEP'
            print(cmd_te)
            o_te = check_output(cmd_te, shell=True, universal_newlines=True)
            print(o_te)



if __name__ == "__main__":
    main()
