
import torch
# from torchvision.models import resnet50, mobilenet_v2
import timm
import os
import torchvision

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