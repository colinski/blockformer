import torch

if __name__ == '__main__':
    data = torch.load('checkpoints/resnet50_8xb256-rsb-a1-600e_in1k_20211228-20e21305.pth', 
            map_location='cpu')
    trained_sd = data['state_dict']
    new_sd = {}
    new_sd['stem.0.weight'] = trained_sd['backbone.conv1.weight']
    new_sd['stem.1.weight'] = trained_sd['backbone.bn1.weight']
    new_sd['stem.1.bias'] = trained_sd['backbone.bn1.weight']
    new_sd['stem.1.running_mean'] = trained_sd['backbone.bn1.running_mean']
    new_sd['stem.1.running_var'] = trained_sd['backbone.bn1.running_var']
    new_sd['stem.1.num_batches_tracked'] = trained_sd['backbone.bn1.num_batches_tracked']
    torch.save(new_sd, 'resnet50_stem.pth')
