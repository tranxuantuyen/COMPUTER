## path line 177, test line 171
#%%
import h5py
from slowfast.config.defaults import assert_and_infer_cfg
from slowfast.utils.misc import launch_job
from slowfast.utils.parser import load_config, parse_args
from tqdm import tqdm
import pathlib
import argparse
from slowfast.models.video_model_builder import MViT
import numpy as np
import h5py
import torch.nn as nn
import torch
from slowfast.utils.checkpoint import load_checkpoint
from slowfast.datasets import loader
def save_dict_to_hdf5(dic, filename):
    """
    ....
    """
    with h5py.File(filename, 'w') as h5file:
        recursively_save_dict_contents_to_group(h5file, '/', dic)

def recursively_save_dict_contents_to_group(h5file, path, dic):
    """
    ....
    """
    for key, item in dic.items():
        if isinstance(item, (np.ndarray, np.int64, np.float64, str, bytes)):
            h5file[path + key] = item
        elif isinstance(item, dict):
            recursively_save_dict_contents_to_group(h5file, path + key + '/', item)
        else:
            raise ValueError('Cannot save %s type'%type(item))

def load_dict_from_hdf5(filename):
    """
    ....
    """
    with h5py.File(filename, 'r') as h5file:
        return recursively_load_dict_contents_from_group(h5file, '/')

def recursively_load_dict_contents_from_group(h5file, path):
    """
    ....
    """
    ans = {}
    for key, item in h5file[path].items():
        if isinstance(item, h5py._hl.dataset.Dataset):
            ans[key] = item[()]
        elif isinstance(item, h5py._hl.group.Group):
            ans[key] = recursively_load_dict_contents_from_group(h5file, path + key + '/')
    return ans

parser = argparse.ArgumentParser(
    description="Provide SlowFast video training and testing pipeline."
)
parser.add_argument(
    "--shard_id",
    help="The shard id of current node, Starts from 0 to num_shards - 1",
    default=0,
    type=int,
)
parser.add_argument(
    "--num_shards",
    help="Number of shards using by the job",
    default=1,
    type=int,
)
parser.add_argument(
    "--init_method",
    help="Initialization method, includes TCP or shared file-system",
    default="tcp://localhost:9999",
    type=str,
)
parser.add_argument(
    "--cfg",
    dest="cfg_file",
    help="Path to the config file",
    default="configs/extract_clip_selection.yaml",
    type=str,
)
parser.add_argument(
    "--mode",
    dest="mode",
    help="train or test",
    default="train",
    type=str,
)
parser.add_argument(
    "--feature_path",
    dest="feature_path",
    help="path to save the extracted feature",
    default=".",
    type=str,
)
parser.add_argument(
    "--backbone_model",
    dest="backbone_path",
    help="path to save the extracted feature",
    default=".",
    type=str,
)
parser.add_argument(
    "opts",
    help="See slowfast/config/defaults.py for all options",
    default=None,
    nargs=argparse.REMAINDER,
)
args = parser.parse_args()
print("config files: {}".format(args))
cfg = load_config(args)
cfg = assert_and_infer_cfg(cfg)

device = 'cuda:0'



# %%
class dummyFunc(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    def forward(self, x):
        return x
class customMvit(MViT):
    def __init__(self, cfg) -> None:
        super(customMvit, self).__init__(cfg)
        self.head.projection = dummyFunc()
        self.head.act = dummyFunc()
    def forward(self, x, video_names=None, bboxes=None):
        x = x[0]
        H = x.shape[3] // self.patch_stride[1]

        x = self.patch_embed(x)[0]
        # print("ajdfjladfjd   ", x[0].shape)
        T = self.cfg.DATA.NUM_FRAMES // self.patch_stride[0]
        W = x.shape[1] // H // T
        B, N, C = x.shape

        if self.cls_embed_on:
            cls_tokens = self.cls_token.expand(
                B, -1, -1
            )  # stole cls_tokens impl from Phil Wang, thanks
            x = torch.cat((cls_tokens, x), dim=1)

        if self.use_abs_pos:
            if self.sep_pos_embed:
                pos_embed = self.pos_embed_spatial.repeat(
                    1, self.patch_dims[0], 1
                ) + torch.repeat_interleave(
                    self.pos_embed_temporal,
                    self.patch_dims[1] * self.patch_dims[2],
                    dim=1,
                )
                if self.cls_embed_on:
                    pos_embed = torch.cat([self.pos_embed_class, pos_embed], 1)
                x = x + pos_embed
            else:
                x = x + self.pos_embed

        if self.drop_rate:
            x = self.pos_drop(x)

        if self.norm_stem:
            x = self.norm_stem(x)

        thw = [T, H, W]
        for blk_idx, blk in enumerate(self.blocks):
            x, thw = blk(x, thw)

        x = self.norm(x)

        if self.enable_detection:
            if self.cls_embed_on:
                x = x[:, 1:]

            B, _, C = x.shape
            encoder_feature = x.clone()
            x = x.transpose(1, 2).reshape(B, C, thw[0], thw[1], thw[2])
            x = self.head([x], bboxes)
        return x, encoder_feature

model = customMvit(cfg)
model = model.to(device)

epoch = load_checkpoint(args.backbone_path, model, data_parallel=False)
train_loader = loader.construct_loader(cfg, args.mode)
model.eval()

for cur_iter, (inputs, labels,_, _,  meta) in tqdm(enumerate(train_loader)):
    feature = {}
    video_id = str(int(meta['metadata'][0][0]))
    temp_path = '{}/{}/{}'.format(args.feature_path, args.mode, video_id)  
    pathlib.Path('{}/{}'.format(args.feature_path, args.mode)  ).mkdir(exist_ok=True)
    pathlib.Path(temp_path).mkdir(exist_ok=True)
    sec = str(int(meta['metadata'][0][1]))
    for i in range(len(inputs)):
        inputs[i] = inputs[i].to(device)
    meta["boxes"] = meta["boxes"].to(device)
    print(inputs[0].shape)
    preds, encoder_feature = model(inputs, meta["video_name"], meta["boxes"])
    
    if video_id not in feature:
        feature[video_id] = {}
    if sec not in feature[video_id]:
        feature[video_id][sec] = {}

    for key in meta:
        meta[key] = meta[key].cpu().detach().numpy()
    feature[video_id][sec]['meta'] = meta

    feature[video_id][sec]['human_feature'] = preds.cpu().detach().numpy()
    # feature[video_id][sec]['box_location'] = meta['ori_boxes'][:, 1:5]
    feature[video_id][sec]['box_label'] = labels.detach().numpy()

    # feature[video_id][sec][box] => feature of human
    feature[video_id][sec]['context'] = encoder_feature.cpu().detach().numpy()

    # print(cur_iter)
    # if cur_iter > 10:
    #     break
    filename = temp_path + '/' + sec + '.h5'
    save_dict_to_hdf5(feature, filename)
#%%
