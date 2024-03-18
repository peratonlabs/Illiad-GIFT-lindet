

import shutil
import os
import argparse




def proc_dir(inpath, outpath, cls):
    model_dirs = os.listdir(inpath)

    for model_dir in model_dirs:
        src_dir = os.path.join(inpath, model_dir)
        dst_dir = os.path.join(outpath, model_dir + '_' + str(cls))

        shutil.copytree(src_dir, dst_dir)

        with open(os.path.join(dst_dir, 'ground_truth.csv'),'w') as f:
            f.write(str(cls))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--tdc_train_path', type=str, help='Path to TDC 2022 detection train directory.')
    # parser.add_argument('--output_path', type=str)

    args = parser.parse_args()

    output_path = os.path.join(args.tdc_train_path,'models')
    os.makedirs(output_path, exist_ok=True)


    proc_dir(os.path.join(args.tdc_train_path,'clean'), output_path, '0')
    proc_dir(os.path.join(args.tdc_train_path,'trojan'), output_path, '1')
    #
    # proc_dir(inpath, args.output_path, cls)
    #
    #
    #
    # model_dirs = os.listdir(args.tdc_train_path)
    #
    #
    #
    #
    # for model_dir in model_dirs:
    #     src_dir = os.path.join(args.tdc_train_path, model_dir)
    #     dst_dir = os.path.join(args.output_path, model_dir)
    #
    #
    #     shutil.copy(src_dir, dst_dir)
    #
    #     with open(os.path.join(dst_dir,'ground_truth.csv')) as f:
    #         f.write('0')


