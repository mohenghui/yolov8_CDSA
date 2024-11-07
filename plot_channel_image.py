import warnings, argparse, torch
warnings.filterwarnings('ignore')
import numpy as np
import torch.nn as nn
import matplotlib.pylab as plt
from ultralytics.nn.tasks import attempt_load_weights

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-weights', type=str, default='', help='base weights path')
    parser.add_argument('--prune-weights', type=str, default='', help='prune weights path')
    opt = parser.parse_args()
    
    print(f'loading base model from {opt.base_weights}')
    base_model = attempt_load_weights(opt.base_weights, device=torch.device('cpu'))
    print(f'loading prune model from {opt.prune_weights}')
    prune_model = attempt_load_weights(opt.prune_weights, device=torch.device('cpu'))
    
    base_model_state_dict, prune_model_state_dict = base_model.state_dict(), prune_model.state_dict()
    
    base_channels, prune_channels, channels_dis, names = [], [], [], []
    for (base_m_name, base_m), (prune_m_name, prune_m) in zip(base_model.named_modules(), prune_model.named_modules()):
        try:
            if isinstance(base_m, nn.Conv2d):
                names.append(base_m_name.replace('model.model.', ''))
                base_channels.append(base_m.out_channels)
                prune_channels.append(prune_m.out_channels)
                channels_dis.append(base_m.out_channels - prune_m.out_channels)
            elif isinstance(base_m, nn.Linear):
                names.append(base_m_name.replace('model.model.', ''))
                base_channels.append(base_m.out_features)
                prune_channels.append(prune_m.out_features)
                channels_dis.append(base_m.out_features - prune_m.out_features)
        except:
            continue
    base_channels, prune_channels, channels_dis, names = np.array(base_channels), np.array(prune_channels), np.array(channels_dis), np.array(names)
    channels_dis_sort = np.argsort(channels_dis)[::-1]
    
    x = np.arange(len(names))
    plt.figure(figsize=(25, 8))
    plt.xticks(x, names, rotation=90)
    plt.bar(x, base_channels, color='orange', label='base')
    plt.bar(x, prune_channels, color='red', label='prune')
    plt.legend(fontsize=20)
    plt.title('Channel contrast diagram', fontdict={'fontsize':20})
    plt.tight_layout()
    plt.savefig('channels_chart')
    print('save to channels_chart.png')
    
    base_channels, prune_channels, names = base_channels[channels_dis_sort], prune_channels[channels_dis_sort], names[channels_dis_sort]
    x = np.arange(len(names))
    plt.figure(figsize=(25, 8))
    plt.xticks(x, names, rotation=90)
    plt.bar(x, base_channels, color='orange', label='base')
    plt.bar(x, prune_channels, color='red', label='prune')
    plt.legend(fontsize=20)
    plt.title('Channel contrast diagram', fontdict={'fontsize':20})
    plt.tight_layout()
    plt.savefig('channels_chart_sort')
    print('save to channels_chart_sort.png')