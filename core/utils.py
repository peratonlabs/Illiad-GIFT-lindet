
import torch
import torchvision
import collections
import model_factories

def get_arch(model_or_path):
    if type(model_or_path) == str:
        if torch.cuda.is_available():
            model = torch.load(model_or_path)
        else:
            model = torch.load(model_or_path, map_location=torch.device('cpu'))
    else:
        model = model_or_path

    if isinstance(model, model_factories.NerLinearModel):
        numparams = len([p for p in model.parameters()])
        cls = [p for p in model.modules()][1].__class__.__name__ + '_' + str(numparams)
    elif isinstance(model, model_factories.LstmLinearModel) or isinstance(model, model_factories.GruLinearModel) or isinstance \
            (model, model_factories.FCLinearModel):
        params = [p for p in model.parameters()]
        numparams = len(params)
        szparams = params[1].shape[0]
        cls = model.__class__.__name__ + '_' + str(numparams) + '_' + str(szparams)

    elif isinstance(model, torchvision.models.resnet.ResNet):
        params = [p for p in model.parameters()]
        numparams = len(params)
        width = params[4].shape[0]
        cls = model.__class__.__name__ + '_' + str(numparams) + '_' + str(width)

    elif isinstance(model, torchvision.models.shufflenetv2.ShuffleNetV2):
        params = [p for p in model.parameters()]
        numparams = len(params)
        width = params[7].shape[0]
        cls = model.__class__.__name__ + '_' + str(numparams) + '_' + str(width)

    elif isinstance(model, torchvision.models.squeezenet.SqueezeNet):
        params = [p for p in model.parameters()]
        numparams = len(params)
        width = params[1].shape[0]
        cls = model.__class__.__name__ + '_' + str(numparams) + '_' + str(width)

    elif isinstance(model, torch.nn.Module):
        numparams = len([p for p in model.parameters()])
        cls = model.__class__.__name__ + '_' + str(numparams)
    elif isinstance(model, collections.OrderedDict):
        numparams = len(model)
        cls = model.__class__.__name__ + '_' + str(numparams)
    else:
        numparams = len(model['state_dict'])
        cls = model['model'] + '_' + str(numparams)
    # print(cls)
    return cls


def proc_feat_type(feat_type, ref_model):
    if feat_type == 'raw':
        norm_type = None
    elif feat_type == 'cosine_delta' or feat_type == 'cosine':
        norm_type = 'cosine'
    elif feat_type == 'white_delta' or feat_type == 'white':
        norm_type = 'white'
    elif feat_type == 'pnorm_delta' or feat_type == 'pnorm':
        norm_type = 'pnorm'
    else:
        assert False, f"bad feat_type {feat_type}"

    if "_delta" not in feat_type:
        ref_model = None

    return norm_type, ref_model
