
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


def gen_r9_json():
    archlist = [
        "DistilBertForQuestionAnswering_102",
        "DistilBertForSequenceClassification_104",
        "DistilBertForTokenClassification_102",
        "ElectraForQuestionAnswering_201",
        "ElectraForSequenceClassification_203",
        "ElectraForTokenClassification_201",
        "RobertaForQuestionAnswering_199",
        "RobertaForSequenceClassification_201",
        "RobertaForTokenClassification_199",
    ]

    gen_schema("./config/base_schema.json", "./config/r9_metaparameters_schema.json", archlist)
    gen_init_json("./config/base.json", "./config/r9_metaparameters.json", archlist)


def gen_r10_json():
    archlist = ["SSD_71", "FasterRCNN_83"]
    gen_schema("./config/base_schema.json", "./config/r10_metaparameters_schema.json", archlist)
    gen_init_json("./config/base.json", "./config/r10_metaparameters.json", archlist)


def gen_r11_json():
    archlist = ["ResNet_161", "VisionTransformer_152", "MobileNetV2_158"]
    gen_schema("./config/base_schema.json", "./config/r11_metaparameters_schema.json", archlist)
    gen_init_json("./config/base.json", "./config/r11_metaparameters.json", archlist)


def gen_r13_json():
    archlist = ["SSD_71", "FasterRCNN_209", "DetrForObjectDetection_326"]
    gen_schema("./config/base_schema.json", "./config/r13_metaparameters_schema.json", archlist)
    gen_init_json("./config/base.json", "./config/r13_metaparameters.json", archlist)


def gen_r14_json():
    archlist = ["SimplifiedRLStarter_18", "BasicFCModel_12"]
    gen_schema("./config/base_schema.json", "./config/r14_metaparameters_schema.json", archlist)
    gen_init_json("./config/base.json", "./config/r14_metaparameters.json", archlist)


def gen_r15_json():
    archlist = ["RobertaForQuestionAnswering_103", "RobertaForQuestionAnswering_199", "MobileBertForQuestionAnswering_1113"]
    gen_schema("./config/base_schema.json", "./config/r15_metaparameters_schema.json", archlist)
    gen_init_json("./config/base.json", "./config/r15_metaparameters.json", archlist)



# gen_schema("./config/base_schema.json", "./config/test_schema.json", ["SSD_71", "FasterRCNN_83"])
# gen_init_json("./config/base.json", "./config/test.json", ["SSD_71", "FasterRCNN_83"])
# validate("./config/test.json", "./config/test_schema.json")
# print(recover_actual_config("./config/test.json"))




