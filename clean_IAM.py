import os
import sys
from PIL import Image
import pandas as pd
import numpy as np
import json
import hw_utils

def main():

    # Configuration file path, passed by arguments or default "./config.json".
    if len(sys.argv) == 1:
        print("Execution without arguments, config file by default: ./config.json")
        config_file=str('./config.json')
    elif len(sys.argv) == 2:
        print("Execution with arguments, config file:" +str(sys.argv[1]))
        config_file = str(sys.argv[1])
    else:
        print()
        print("ERROR")
        print("Wrong number of arguments. Execute:")
        print(">> python3 clean_IAM.py [path_config_file]")
        exit(1)


    # We load the configuration file
    try:
        data = json.load(open(config_file))
    except FileNotFoundError:
        print()
        print("ERROR")
        print("No such config file : " + config_file)
        exit(1)


    # If the destination directory does not exist, it is created.
    if not os.path.exists(str(data["general"]["processed_data_path"])):
        os.mkdir(str(data["general"]["processed_data_path"]))

    # list with all the files in the directory.
    lstDir = os.walk(str(data["general"]["raw_data_path"]))

    #The names of the suitable images are read from the csv.
    df = pd.read_csv(str(data["general"]["csv_path"]), sep=",",index_col="index")
    df = df.loc[:, ['image']]
    lstIm = df.as_matrix()

    # We go through the directory where the IAM images are stored.
    for root, dirs, files in lstDir:
        for file in files:
            (name, ext) = os.path.splitext(file)
            # We check if the file is in the apt image array.
            if name in lstIm:
                #Function "scale_invert" is executed
                hw_utils.scale_invert(str(root)+str("/")+str(name+ext),
                str(data["general"]["processed_data_path"])+str(name+ext),int(data["general"]["height"]),int(data["general"]["width"]))


if __name__ == "__main__":
    main()
