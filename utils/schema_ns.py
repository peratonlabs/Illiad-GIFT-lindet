
import json
import copy


def gen_field_name(arch, base_name):
    return "train_" + "$" + arch + "$" + base_name


def parse_field_name(field_name):
    ns, arch, base_name = field_name.split("$")
    return arch, base_name


def recover_actual_config(metaparameters_filepath):
    with open(metaparameters_filepath) as config_file:
        config_json = json.load(config_file)

    actual_config = {}

    for k, v in config_json.items():
        arch, field_name = parse_field_name(k)
        if arch not in actual_config:
            actual_config[arch] = {}
        actual_config[arch][field_name] = v

    return actual_config


def gen_schema(base_schema_path, output_path, archlist):
    # replace properties with flattened dict of properties

    with open(base_schema_path) as f:
        base_schema = json.load(f)

    new_schema = copy.deepcopy(base_schema)
    new_schema['properties'] = {}

    for arch in archlist:
        for k, v in base_schema['properties'].items():
            new_k = gen_field_name(arch, k)
            new_schema['properties'][new_k] = v

    new_schema['required'] = [k for k in new_schema['properties'].keys()]

    with open(output_path, "w") as wf:
        json.dump(new_schema, wf, indent=4)

def gen_init_json(base_json_path, output_path, archlist):
    # replace properties with flattened dict of properties

    with open(base_json_path) as f:
        base_json = json.load(f)

    new_json = {}
    for arch in archlist:
        for k, v in base_json.items():
            new_k = gen_field_name(arch, k)
            new_json[new_k] = v

    with open(output_path, "w") as wf:
        json.dump(new_json, wf, indent=4)


def validate(metaparameters_filepath, schema_filepath):
    import jsonschema

    with open(metaparameters_filepath) as config_file:
        config_json = json.load(config_file)
    with open(schema_filepath) as schema_file:
        schema_json = json.load(schema_file)

    jsonschema.validate(instance=config_json, schema=schema_json)
    print("validation complete")


# def gen_r9_json():
#     archlist = [
#         "DistilBertForQuestionAnswering_102",
#         "DistilBertForSequenceClassification_104",
#         "DistilBertForTokenClassification_102",
#         "ElectraForQuestionAnswering_201",
#         "ElectraForSequenceClassification_203",
#         "ElectraForTokenClassification_201",
#         "RobertaForQuestionAnswering_199",
#         "RobertaForSequenceClassification_201",
#         "RobertaForTokenClassification_199",
#     ]

#     gen_schema("./config/base_schema.json", "./config/r9_metaparameters_schema.json", archlist)
#     gen_init_json("./config/base.json", "./config/r9_metaparameters.json", archlist)


# def gen_r10_json():
#     archlist = ["SSD_71", "FasterRCNN_83"]
#     gen_schema("./config/base_schema.json", "./config/r10_metaparameters_schema.json", archlist)
#     gen_init_json("./config/base.json", "./config/r10_metaparameters.json", archlist)


# def gen_r11_json():
#     archlist = ["ResNet_161", "VisionTransformer_152", "MobileNetV2_158"]
#     gen_schema("./config/base_schema.json", "./config/r11_metaparameters_schema.json", archlist)
#     gen_init_json("./config/base.json", "./config/r11_metaparameters.json", archlist)


# def gen_r13_json():
#     archlist = ["SSD_71", "FasterRCNN_209", "DetrForObjectDetection_326"]
#     gen_schema("./config/base_schema.json", "./config/r13_metaparameters_schema.json", archlist)
#     gen_init_json("./config/base.json", "./config/r13_metaparameters.json", archlist)


# def gen_r14_json():
#     archlist = ["SimplifiedRLStarter_18", "BasicFCModel_12"]
#     gen_schema("./config/base_schema.json", "./config/r14_metaparameters_schema.json", archlist)
#     gen_init_json("./config/base.json", "./config/r14_metaparameters.json", archlist)


# def gen_r15_json():
#     archlist = ["RobertaForQuestionAnswering_103", "RobertaForQuestionAnswering_199", "MobileBertForQuestionAnswering_1113"]
#     gen_schema("./config/base_schema.json", "./config/r15_metaparameters_schema.json", archlist)
#     gen_init_json("./config/base.json", "./config/r15_metaparameters.json", archlist)


# def gen_r8_json():
#     archlist = [
#         "ElectraForQuestionAnswering_201",
#         "RobertaForQuestionAnswering_199",
#     ]

#     gen_schema("./config/base_schema.json", "./config/r8_metaparameters_schema.json", archlist)
#     gen_init_json("./config/base.json", "./config/r8_metaparameters.json", archlist)


# def gen_r7_json():
#     archlist = [
#         "MobileBertModel_1113",
#         "BertModel_199",
#         "RobertaModel_199",
#         "DistilBertModel_102",
#     ]

#     gen_schema("./config/base_schema.json", "./config/r7_metaparameters_schema.json", archlist)
#     gen_init_json("./config/base.json", "./config/r7_metaparameters.json", archlist)


# def gen_r6_json():
#     archlist = ['FCLinearModel_6_256', 'LstmLinearModel_18_1024', 'GruLinearModel_10_768', 'LstmLinearModel_34_1024', 'GruLinearModel_34_1536', 'LstmLinearModel_18_2048', 'FCLinearModel_10_256', 'FCLinearModel_10_512', 'GruLinearModel_18_768', 'FCLinearModel_6_512', 'LstmLinearModel_34_2048', 'GruLinearModel_18_1536', 'GruLinearModel_10_1536', 'LstmLinearModel_10_1024', 'LstmLinearModel_10_2048', 'GruLinearModel_34_768']

#     gen_schema("./config/base_schema.json", "./config/r6_metaparameters_schema.json", archlist)
#     gen_init_json("./config/base.json", "./config/r6_metaparameters.json", archlist)

# def gen_r5_json():
#     archlist = ['GruLinearModel_18_768', 'LstmLinearModel_18_1024', 'LinearModel_2']

#     gen_schema("./config/base_schema.json", "./config/r5_metaparameters_schema.json", archlist)
#     gen_init_json("./config/base.json", "./config/r5_metaparameters.json", archlist)

# def gen_r4_json():
#     archlist = []

#     gen_schema("./config/base_schema.json", "./config/r4_metaparameters_schema.json", archlist)
#     gen_init_json("./config/base.json", "./config/r4_metaparameters.json", archlist)

# def gen_r3_json():
#     archlist = []

#     gen_schema("./config/base_schema.json", "./config/r3_metaparameters_schema.json", archlist)
#     gen_init_json("./config/base.json", "./config/r3_metaparameters.json", archlist)

# def gen_r2_json():
#     archlist = []

#     gen_schema("./config/base_schema.json", "./config/r2_metaparameters_schema.json", archlist)
#     gen_init_json("./config/base.json", "./config/r2_metaparameters.json", archlist)

# def gen_r1_json():
#     archlist = ['ResNet_161', 'DenseNet_364', 'Inception3_284']

#     gen_schema("./config/base_schema.json", "./config/r1_metaparameters_schema.json", archlist)
#     gen_init_json("./config/base.json", "./config/r1_metaparameters.json", archlist)

# gen_schema("./config/base_schema.json", "./config/test_schema.json", ["SSD_71", "FasterRCNN_83"])
# gen_init_json("./config/base.json", "./config/test.json", ["SSD_71", "FasterRCNN_83"])
# validate("./config/test.json", "./config/test_schema.json")
# print(recover_actual_config("./config/test.json"))

arch_lists = {
    1: ['ResNet_161', 'DenseNet_364', 'Inception3_284'],
    2: [],
    3: [],
    4: [],
    5: ['GruLinearModel_18_768', 'LstmLinearModel_18_1024', 'LinearModel_2'],
    6: ['FCLinearModel_6_256', 'LstmLinearModel_18_1024', 'GruLinearModel_10_768', 'LstmLinearModel_34_1024', 'GruLinearModel_34_1536', 'LstmLinearModel_18_2048', 'FCLinearModel_10_256', 'FCLinearModel_10_512', 'GruLinearModel_18_768', 'FCLinearModel_6_512', 'LstmLinearModel_34_2048', 'GruLinearModel_18_1536', 'GruLinearModel_10_1536', 'LstmLinearModel_10_1024', 'LstmLinearModel_10_2048', 'GruLinearModel_34_768'],
    7: ["MobileBertModel_1113","BertModel_199","RobertaModel_199","DistilBertModel_102"],
    8: ["ElectraForQuestionAnswering_201","RobertaForQuestionAnswering_199"],
    9: ["DistilBertForQuestionAnswering_102","DistilBertForSequenceClassification_104","DistilBertForTokenClassification_102", "ElectraForQuestionAnswering_201","ElectraForSequenceClassification_203","ElectraForTokenClassification_201","RobertaForQuestionAnswering_199","RobertaForSequenceClassification_201","RobertaForTokenClassification_199"],
    10: ["SSD_71", "FasterRCNN_83"],
    11: ["ResNet_161", "VisionTransformer_152", "MobileNetV2_158"],
    12: [],
    13: ["SSD_71", "FasterRCNN_209", "DetrForObjectDetection_326"],
    14: ["SimplifiedRLStarter_18", "BasicFCModel_12"],
    15: ["RobertaForQuestionAnswering_103", "RobertaForQuestionAnswering_199", "MobileBertForQuestionAnswering_1113"]
}


def gen_exp(round, base="./config/base.json", suffix=""):
    archlist = arch_lists[round]
    # gen_schema("./config/base_schema.json", "./config/r1_metaparameters_schema.json", archlist)
    gen_init_json(base, "./config/r" + str(round) + suffix + "_metaparameters.json", archlist)


def gen_round_schema(round):
    archlist = arch_lists[round]
    gen_schema("./config/base_schema.json", "./config/r" + str(round) + "_metaparameters_schema.json", archlist)


def gen_exp_configs():
    base_files = ["./config/base.json", "./config/baseA.json", "./config/baseB.json", "./config/baseC.json", "./config/baseD.json", "./config/baseE.json", "./config/baseF.json"]
    names = ["", "A", "B", "C", "D", "E", "F"]
    rounds = [k for k in arch_lists.keys()]

    for round in rounds:
        for name, base_file in zip(names, base_files):
            gen_exp(round, base=base_file, suffix=name)





