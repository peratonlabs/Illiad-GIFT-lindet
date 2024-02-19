# Linear Trojan Detection



## Setup

Create a suitable conda environment (see [Multiround Environments](Multiround Environments) for more examples)

```
conda create -n rXX python=3.8.17
conda activate rXX
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
pip install transformers==4.32.1
pip install opencv-python==4.8.0.74
pip install jsonargparse==4.23.0 jsonschema==4.18.4
pip install scikit-learn==1.3.0 scipy==1.10.1
pip install timm==0.9.7
```

Automatically generate schemas and baseline metaparameter configs.
```
python -m utils.schema_ns
```


## Examples Usages

Example calibration (outside singularity): 
```
python wa_detector.py --configure_mode --gift_basepath ./ --configure_models_dirpath /path/to/round18 --scratch_dirpath ./scratch/ --num_cv_trials 10 --metaparameters_filepath ./config/final/r18SB_metaparameters.json --schema_filepath ./config/r18_metaparameters_schema.json --round 18 --learned_parameters_dirpath learned_parameters/round18SB
```

Example inference (outside singularity): 
```
python wa_detector.py --gift_basepath ./ --model_filepath /path/to/round18/models/id-XXXXXXXX/model.pt --result_filepath ./scratch/output.txt --scratch_dirpath ./scratch/ --metaparameters_filepath ./config/final/r18SB_metaparameters.json --schema_filepath ./config/r18_metaparameters_schema.json --round 18 --learned_parameters_dirpath learned_parameters/round18SB --examples_dirpath ./123fakepath/
```

Example container build: 
Make sure the paths in the .def file are accurate.
```
sudo singularity build --force ./containers/cyber-network-c2-feb2024_sts_coslin.simg ./singularity/r18SB.def
```

Example container execution: 
```
singularity run --nv ./containers/cyber-network-c2-feb2024_sts_coslin.simg --model_filepath=./path/to/round18/models/id-XXXXXXXX/model.pt --result_filepath=./scratch/output.txt --scratch_dirpath=./scratch/ --metaparameters_filepath=/metaparameters.json --schema_filepath=/metaparameters_schema.json --learned_parameters_dirpath=/learned_parameters/ --examples_dirpath=./123fakepath/
```

## Adding a round
To apply this method to a new round, you need to perform the following steps:

### Run wa_detector
```
python wa_detector.py --configure_mode --gift_basepath ./ --configure_models_dirpath /path/to/roundXX --scratch_dirpath ./scratch/trial1 --num_cv_trials 10 --metaparameters_filepath ./config/r11_metaparameters.json --schema_filepath ./config/r11_metaparameters.json --round 11
```
This will crash, but it and find the list of architecture names & number of tensors (e.g., RobertaForQuestionAnswering_103).

Note that each round may have its own environment issues, which could cause this script to crash prior to printing the architectures.

### Generate schema and json config using schema_ns utility
```
import utils.schema_ns
archlist = ["RobertaForQuestionAnswering_103", "RobertaForQuestionAnswering_199", "MobileBertForQuestionAnswering_1113"]
utils.schema_ns.gen_schema("./config/base_schema.json", "./config/rXX_metaparameters_schema.json", archlist)
utils.schema_ns.gen_init_json("./config/base.json", "./config/rXX_metaparameters.json", archlist)
```

### Build a reference model function (optional) 
To maximize performance on many rounds, we identify the pretrained source model for each architecture. 

This is currently handled with a big if-else statement around line 64 in wa_detector.py. 

If you would like to use reference models, we recommend the following design pattern:
```
elif  args.round == 'XX':
        ref_models.rXX_check_for_ref_models(model_dir)
        ref_model_function = lambda arch: ref_models.rXX_load_ref_model(arch, model_dir)
```
The script first downloads any required models with rXX_check_for_ref_models, then builds a function that loads the reference model for each architecture with rXX_load_ref_model. utils.ref_models has several examples. 

Alternatively, you can skip this step and ignore the reference models. This is recommended for the first pass through a new round. 

### Calibrate the wa_detector
Modify rXX_metaparameters.json as desired (defaults are reasonable) and rerun calibration.

```
python wa_detector.py --configure_mode --gift_basepath ./ --configure_models_dirpath /path/to/roundXX --scratch_dirpath ./scratch/trial1 --num_cv_trials 10 --metaparameters_filepath ./config/rXX_metaparameters.json --schema_filepath ./config/rXX_metaparameters.json --round XX
```

Note that each round may have its own environment issues, which could cause this script to crash.

This will run 10 cross validation trials, then calibrate on the full dataset. To just quickly calibrate, set num_cv_trials to 0 or 1. 

### Create a singularity file
Several examples are available in the ./singularity directory. Be sure to update the exec command for the current round. Different rounds may have different environment requirements.

### Build & execute singularity container
See above.


## Multiround Environments


### Round XX env
Conda and pytorch are struggling to work together, so I recommend setting up the env as follows:
```
conda create -n r11XX python=3.8.17
conda activate r11XX
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
pip install transformers==4.32.1
pip install opencv-python==4.8.0.74
pip install jsonargparse==4.23.0 jsonschema==4.18.4
pip install scikit-learn==1.3.0 scipy==1.10.1
pip install timm==0.9.7
```
This environment should work for rounds 1, 5-12, 15, 18

This environment may work on r11 if you run pip install timm==0.6.13 instead of 0.9.7. 


### Round 14 env
Conda and pytorch are struggling to work together, so I recommend setting up the env as follows:
```
conda create -n r14new python=3.8.17
conda activate r14new
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
pip install transformers==4.32.1
pip install opencv-python==4.8.0.74
pip install jsonargparse==4.23.0 jsonschema==4.18.4
pip install scikit-learn==1.3.0 scipy==1.10.1
pip install timm==0.9.7
pip install trojai_rl
```
This environment should work for round 14

### Round 13 env
Conda and pytorch are struggling to work together, so I recommend setting up the env as follows:
```
conda create -n r13new python=3.8.17
conda activate r13new
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116
pip install timm transformers==4.23.1 jsonschema jsonargparse jsonpickle scikit-learn scikit-image
```
This environment should work for round 13

### Round 4 env

```
conda create -n r4new python=3.8
conda activate r4new
pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
pip install joblib
pip install jsonargparse==4.23.0 jsonschema==4.18.4
pip install scikit-learn==1.3.0 scipy==1.10.1
pip install timm
```
This environment should work for rounds 2-4

## Round 16 env

```
conda create -n r16new2 python=3.8
conda activate r16new
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install gym gymnasium minigrid jsonschema jsonpickle scikit-learn opencv-python
pip install jsonargparse==4.23.0

```
This environment should work for round 16. (need to get specific versions)
