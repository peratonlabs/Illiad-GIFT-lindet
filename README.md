# cosine-linear method

Example calibration: 
```
python wa_detector.py --configure_mode --gift_basepath ./ --configure_models_dirpath /path/to/round15 --scratch_dirpath /path/to/round15/scratch/trial1 --num_cv_trials 10 --metaparameters_filepath ./config/r15_metaparameters.json --schema_filepath ./config/r15_metaparameters.json --round 15
```

Example container build: 
```
sudo singularity build --force ./containers/nlp-question-answering-aug2023_test_sts_coslin.simg    ./singularity/r15_wa_detector.def
```

Example container execution: 
```
singularity run --nv ./containers/nlp-question-answering-aug2023_test_sts_coslin.simg --model_filepath=./scratch/round15/id-00000000/model.pt --result_filepath=./scratch/output.txt --scratch_dirpath=./scratch/ --metaparameters_filepath=/metaparameters.json --schema_filepath=/metaparameters_schema.json --learned_parameters_dirpath=/learned_parameters/ --examples_dirpath=./123fakepath/
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


## Multiround environments


### Round 11 env

```
conda env create -f r11new2.yml
```
Note: conda seems to have more issues than pip, so I recommend the following for addressing new requirements:
```
conda create -n r11new2 python=3.8.17
pip install <whatever packages>
```




