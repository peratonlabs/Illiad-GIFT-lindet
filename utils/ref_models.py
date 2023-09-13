
import torch
# from torchvision.models import resnet50, mobilenet_v2
import timm
import os
import torchvision


def r13_check_for_ref_models(model_dir):
    # object_detection: ssd300_vgg16
    # object_detection: detr
    # object_detection: fasterrcnn_resnet50_fpn_v2

    # this function is called once during setup
    pth = os.path.join(model_dir, 'FasterRCNN_209.pt')
    if not os.path.exists(pth):
        ref_model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2()
        # torch.save(ref_model.state_dict(), pth)
        torch.save(ref_model, pth)

    pth = os.path.join(model_dir, 'SSD_71.pt')
    if not os.path.exists(pth):
        ref_model = torchvision.models.detection.ssd300_vgg16(weights=torchvision.models.detection.SSD300_VGG16_Weights.DEFAULT)
        # torch.save(ref_model.state_dict(), pth)
        torch.save(ref_model, pth)

    pth = os.path.join(model_dir, 'DetrForObjectDetection_326_r13.pt')
    if not os.path.exists(pth):
        import transformers
        ref_model = transformers.models.detr.modeling_detr.DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")

        # ref_model = torchvision.models.detection.ssd300_vgg16(weights=torchvision.models.detection.SSD300_VGG16_Weights.DEFAULT)
        # torch.save(ref_model.state_dict(), pth)
        torch.save(ref_model, pth)



def r13_load_ref_model(arch, model_dir):
    # 'DetrForObjectDetection_326', 'FasterRCNN_209', 'SSD_71'
    if "FasterRCNN_209" == arch:
        # ref_model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights_backbone=None)
        # ref_model.load_state_dict(torch.load(os.path.join(model_dir, 'frcnn_COCO_V1.pt')))

        ref_model = torch.load(os.path.join(model_dir, 'FasterRCNN_209.pt'))

    elif "SSD_71" == arch:
        # ref_model = torchvision.models.detection.ssd300_vgg16(weights_backbone=None)
        # ref_model.load_state_dict(torch.load(os.path.join(model_dir, 'ssd300_vgg16_coco.pt')))

        ref_model = torch.load(os.path.join(model_dir, 'SSD_71.pt'))
    elif "DetrForObjectDetection_326" == arch:
        # ref_model = torchvision.models.detection.ssd300_vgg16(weights_backbone=None)
        # ref_model.load_state_dict(torch.load(os.path.join(model_dir, 'ssd300_vgg16_coco.pt')))

        ref_model = torch.load(os.path.join(model_dir, 'DetrForObjectDetection_326_r13.pt'))

    else:
        assert False, "bad arch"
    if torch.cuda.is_available():
        ref_model.cuda()
    return ref_model


def r11_check_for_ref_models(model_dir):
    # this function is called once during setup
    pth = os.path.join(model_dir, 'resnet50_V2.pt')
    if not os.path.exists(pth):
        ref_model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2)
        torch.save(ref_model.state_dict(), pth)

    pth = os.path.join(model_dir, 'vit.pt')
    if not os.path.exists(pth):
        ref_model = timm.create_model('vit_base_patch32_224', pretrained=True)  #needs updated probably
        torch.save(ref_model.state_dict(), pth)

    pth = os.path.join(model_dir, 'mobilenet_V2.pt')
    if not os.path.exists(pth):
        ref_model = torchvision.models.mobilenet_v2(weights=torchvision.models.MobileNet_V2_Weights.IMAGENET1K_V2)
        torch.save(ref_model.state_dict(), pth)


def r11_load_ref_model(arch, model_dir):
    # this function is called whenever we get features
    if "ResNet_161" == arch:
        # cfg_dicts[arch] = cfg_dict_resnet
        ref_model = torchvision.models.resnet50()
        ref_model.load_state_dict(torch.load(os.path.join(model_dir, 'resnet50_V2.pt')))
    elif "VisionTransformer_152" == arch:
        # cfg_dicts[arch] = cfg_dict_vit
        ref_model = timm.create_model('vit_base_patch32_224')
        ref_model.load_state_dict(torch.load(os.path.join(model_dir, 'vit.pt')))
    elif "MobileNetV2_158" == arch:
        # cfg_dicts[arch] = cfg_dict_mobilenet
        ref_model = torchvision.models.mobilenet_v2()
        ref_model.load_state_dict(torch.load(os.path.join(model_dir, 'mobilenet_V2.pt')))
    else:
        assert False, "bad arch"
    if torch.cuda.is_available():
        ref_model.cuda()
    return ref_model


def r10_check_for_ref_models(model_dir):
    # this function is called once during setup
    pth = os.path.join(model_dir, 'frcnn_COCO_V1.pt')
    if not os.path.exists(pth):
        ref_model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.COCO_V1)
        # torch.save(ref_model.state_dict(), pth)
        torch.save(ref_model, pth)

    pth = os.path.join(model_dir, 'ssd300_vgg16_coco.pt')
    if not os.path.exists(pth):
        ref_model = torchvision.models.detection.ssd300_vgg16(weights=torchvision.models.detection.SSD300_VGG16_Weights.DEFAULT)
        # torch.save(ref_model.state_dict(), pth)
        torch.save(ref_model, pth)



def r10_load_ref_model(arch, model_dir):

    if "FasterRCNN_83" == arch:
        # ref_model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights_backbone=None)
        # ref_model.load_state_dict(torch.load(os.path.join(model_dir, 'frcnn_COCO_V1.pt')))

        ref_model = torch.load(os.path.join(model_dir, 'frcnn_COCO_V1.pt'))

    elif "SSD_71" == arch:
        # ref_model = torchvision.models.detection.ssd300_vgg16(weights_backbone=None)
        # ref_model.load_state_dict(torch.load(os.path.join(model_dir, 'ssd300_vgg16_coco.pt')))

        ref_model = torch.load(os.path.join(model_dir, 'ssd300_vgg16_coco.pt'))

    else:
        assert False, "bad arch"
    if torch.cuda.is_available():
        ref_model.cuda()
    return ref_model






def r9_check_for_ref_models(model_dir):
    import transformers
    # this function is called once during setup

    pth = os.path.join(model_dir, 'electra_qa.pt')
    if not os.path.exists(pth):
        ref_model = transformers.models.electra.modeling_electra.ElectraForQuestionAnswering.from_pretrained(
            'google/electra-small-discriminator')
        torch.save(ref_model, pth)

    pth = os.path.join(model_dir, 'electra_sc.pt')
    if not os.path.exists(pth):
        ref_model = transformers.models.electra.modeling_electra.ElectraForSequenceClassification.from_pretrained(
            'google/electra-small-discriminator')
        torch.save(ref_model, pth)

    pth = os.path.join(model_dir, 'electra_ner.pt')
    if not os.path.exists(pth):
        ref_model = transformers.models.electra.modeling_electra.ElectraForTokenClassification.from_pretrained('google/electra-small-discriminator')
        torch.save(ref_model, pth)

    pth = os.path.join(model_dir, 'distilbert_qa.pt')
    if not os.path.exists(pth):
        ref_model = transformers.models.distilbert.modeling_distilbert.DistilBertForQuestionAnswering.from_pretrained(
            'distilbert-base-cased')
        torch.save(ref_model, pth)

    pth = os.path.join(model_dir, 'distilbert_sc.pt')
    if not os.path.exists(pth):
        ref_model = transformers.models.distilbert.modeling_distilbert.DistilBertForSequenceClassification.from_pretrained(
            'distilbert-base-cased')
        torch.save(ref_model, pth)

    pth = os.path.join(model_dir, 'distilbert_ner.pt')
    if not os.path.exists(pth):
        ref_model = transformers.models.distilbert.modeling_distilbert.DistilBertForTokenClassification.from_pretrained(
            'distilbert-base-cased')
        torch.save(ref_model, pth)

    pth = os.path.join(model_dir, 'roberta_qa.pt')
    if not os.path.exists(pth):
        ref_model = transformers.models.roberta.modeling_roberta.RobertaForQuestionAnswering.from_pretrained(
            'roberta-base')
        torch.save(ref_model, pth)

    pth = os.path.join(model_dir, 'roberta_sc.pt')
    if not os.path.exists(pth):
        ref_model = transformers.models.roberta.modeling_roberta.RobertaForSequenceClassification.from_pretrained(
            'roberta-base')
        torch.save(ref_model, pth)

    pth = os.path.join(model_dir, 'roberta_ner.pt')
    if not os.path.exists(pth):
        ref_model = transformers.models.roberta.modeling_roberta.RobertaForTokenClassification.from_pretrained('roberta-base')
        torch.save(ref_model, pth)


def r9_load_ref_model(arch, model_dir):

    fnmap = {
        "DistilBertForQuestionAnswering_102": 'distilbert_qa.pt',
        "DistilBertForSequenceClassification_104": 'distilbert_sc.pt',
        "DistilBertForTokenClassification_102": 'distilbert_ner.pt',
        "ElectraForQuestionAnswering_201": 'electra_qa.pt',
        "ElectraForSequenceClassification_203": 'electra_sc.pt',
        "ElectraForTokenClassification_201": 'electra_ner.pt',
        "RobertaForQuestionAnswering_199": 'roberta_qa.pt',
        "RobertaForSequenceClassification_201": 'roberta_sc.pt',
        "RobertaForTokenClassification_199": 'roberta_ner.pt',
    }

    if arch in  fnmap:
        ref_model = torch.load(os.path.join(model_dir, fnmap[arch]))
    else:
        assert False, "bad arch"
    if torch.cuda.is_available():
        ref_model.cuda()
    return ref_model


def r8_check_for_ref_models(model_dir):
    import transformers
    # this function is called once during setup

    pth = os.path.join(model_dir, 'electra_qa.pt')
    if not os.path.exists(pth):
        ref_model = transformers.models.electra.modeling_electra.ElectraForQuestionAnswering.from_pretrained(
            'google/electra-small-discriminator')
        torch.save(ref_model, pth)

    pth = os.path.join(model_dir, 'roberta_qa.pt')
    if not os.path.exists(pth):
        ref_model = transformers.models.roberta.modeling_roberta.RobertaForQuestionAnswering.from_pretrained(
            'roberta-base')
        torch.save(ref_model, pth)


def r8_load_ref_model(arch, model_dir):

    fnmap = {
        "ElectraForQuestionAnswering_201": 'electra_qa.pt',
        "RobertaForQuestionAnswering_199": 'roberta_qa.pt',
    }

    if arch in  fnmap:
        ref_model = torch.load(os.path.join(model_dir, fnmap[arch]))
    else:
        assert False, "bad arch"
    if torch.cuda.is_available():
        ref_model.cuda()
    return ref_model


def r7_check_for_ref_models(model_dir):
    import transformers
    # this function is called once during setup

    # transformers.models.mobilebert.modeling_mobilebert.MobileBertModel   id 0 1113 params       model.modules
    # transformers.models.bert.modeling_bert.BertModel                     id 1 199 params
    # transformers.models.roberta.modeling_roberta.RobertaModel            id 2 199 params
    # transformers.models.distilbert.modeling_distilbert.DistilBertModel   id 6 102 params
    


    pth = os.path.join(model_dir, 'mobilebert_r7.pt')
    if not os.path.exists(pth):
        ref_model = transformers.models.mobilebert.modeling_mobilebert.MobileBertModel.from_pretrained(
            'google/mobilebert-uncased')
        torch.save(ref_model, pth)

    pth = os.path.join(model_dir, 'bert_r7.pt')
    if not os.path.exists(pth):
        ref_model = transformers.models.bert.modeling_bert.BertModel.from_pretrained(
            'bert-base-uncased')
        torch.save(ref_model, pth)
    
    pth = os.path.join(model_dir, 'roberta_r7.pt')
    if not os.path.exists(pth):
        ref_model = transformers.models.roberta.modeling_roberta.RobertaModel.from_pretrained(
            'roberta-base')
        torch.save(ref_model, pth)
    
    pth = os.path.join(model_dir, 'distillbert_r7.pt')
    if not os.path.exists(pth):
        ref_model = transformers.models.distilbert.modeling_distilbert.DistilBertModel.from_pretrained(
            'distilbert-base-cased')
        torch.save(ref_model, pth)


def r7_load_ref_model(arch, model_dir):

    fnmap = {
        "MobileBertModel_1113": 'mobilebert_r7.pt',
        "BertModel_199": 'bert_r7.pt',
        "RobertaModel_199": 'roberta_r7.pt',
        "DistilBertModel_102": 'distillbert_r7.pt',
    }

    if arch in  fnmap:
        ref_model = torch.load(os.path.join(model_dir, fnmap[arch]))
    else:
        assert False, "bad arch"
    if torch.cuda.is_available():
        ref_model.cuda()
    return ref_model


def r15_check_for_ref_models(model_dir):
    import transformers
    # this function is called once during setup

    pth = os.path.join(model_dir, 'tinyroberta15_qa.pt')
    if not os.path.exists(pth):
        ref_model = transformers.models.roberta.modeling_roberta.RobertaForQuestionAnswering.from_pretrained(
            'deepset/tinyroberta-squad2')
        torch.save(ref_model, pth)

    pth = os.path.join(model_dir, 'roberta15_qa.pt')
    if not os.path.exists(pth):
        ref_model = transformers.models.roberta.modeling_roberta.RobertaForQuestionAnswering.from_pretrained(
            'deepset/roberta-base-squad2')
        torch.save(ref_model, pth)

    pth = os.path.join(model_dir, 'mobilebert15_qa.pt')
    if not os.path.exists(pth):
        ref_model = transformers.models.mobilebert.modeling_mobilebert.MobileBertForQuestionAnswering.from_pretrained(
            'csarron/mobilebert-uncased-squad-v2')
        torch.save(ref_model, pth)


def r15_load_ref_model(arch, model_dir):

    fnmap = {
        "RobertaForQuestionAnswering_103": 'tinyroberta15_qa.pt',
        "RobertaForQuestionAnswering_199": 'roberta15_qa.pt',
        "MobileBertForQuestionAnswering_1113": 'mobilebert15_qa.pt',
    }

    if arch in  fnmap:
        ref_model = torch.load(os.path.join(model_dir, fnmap[arch]))
    else:
        assert False, "bad arch"
    if torch.cuda.is_available():
        ref_model.cuda()
    return ref_model
