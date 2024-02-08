

import os
import json
from jsonargparse import ArgumentParser, ActionConfigFile



if __name__ == "__main__":

    
    parser = ArgumentParser()
    parser.add_argument('--inpth', type=str)
    parser.add_argument('--injson', type=str)
    parser.add_argument('--outjson', type=str)

    args = parser.parse_args()



    with open(args.inpth) as f:
        lines = f.readlines()

    arch_dict = {}
    for line in lines:
        line = line[:-1]
        tokens = line.split(' ')


        # print(tokens)

        arch = tokens[0]
        C=tokens[3]
        ce=tokens[-1]

        if arch not in arch_dict or ce<arch_dict[arch][1]:
            arch_dict[arch] = (C, ce)

        # if ce<arch_dict[arch][1]:
        #     arch_dict[arch] = (C, ce)
        # print(arch, C, ce)

    # print(arch_dict)


    with open(args.injson) as f:
        mp = json.load(f)


    for arch in arch_dict.keys():
        field = 'train_$' + arch + '$C'
        
        mp[field] = float(arch_dict[arch][0])

        field = 'train_$' + arch + '$param_batch_sz'
        mp[field]=20



    with open(args.outjson, "w") as f:
        json.dump(mp, f, indent=4)


