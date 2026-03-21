import json
import os
from pathlib import Path
import time
import numpy as np
import pytorch_lightning as pl
import torch
# from rich import get_console
# from rich.table import Table
from omegaconf import OmegaConf
# import moviepy.editor as mp
from tqdm import tqdm
from motGPT.config import parse_args
from motGPT.data.build_data import build_data
from motGPT.models.build_model import build_model
from motGPT.utils.logger import create_logger
import motGPT.render.matplot.plot_3d_global as plot_3d

def motion_token_to_string(motion_token, lengths, codebook_size=512):
    motion_string = []
    for i in range(motion_token.shape[0]):
        motion_i = motion_token[i].cpu(
        ) if motion_token.device.type == 'cuda' else motion_token[i]
        motion_list = motion_i.tolist()[:lengths[i]]
        motion_string.append(
            (f'<motion_id_{codebook_size}>' +
             ''.join([f'<motion_id_{int(i)}>' for i in motion_list]) +
             f'<motion_id_{codebook_size + 1}>'))
    return motion_string


def load_example_input(txt_path, task, model):
    with open(txt_path, "r") as file:
        Lines = file.readlines()
    Lines = [line for line in Lines if line.strip()]
    count = 0
    texts = []
    # Strips the newline character
    motion_joints = [torch.zeros((1, 1, 22, 3))] * len(Lines)
    motion_lengths = [0] * len(Lines)
    motion_token_string = ['']
    motion_head = []
    motion_heading = []
    motion_tailing = []
    motion_feats = []
    # motion_token = torch.zeros((1, 263))
    input_motion_holder_seq = model.lm.input_motion_holder_seq
    output_motion_holder_seq = model.lm.output_motion_holder_seq
    masked_holder_seq = input_motion_holder_seq+model.lm.masked_holder_seq+input_motion_holder_seq

    for i, line in enumerate(Lines):
        count += 1
        splits = line.split('#')
        text = splits[0]
        if len(splits) > 1:
            feat_path = splits[1].replace('\n', '')
            feat_exist = os.path.exists(feat_path)
            if 'motion_placeholder' in text:
                assert feat_exist, FileNotFoundError(feat_path)
            if not feat_exist: continue
            feats = torch.tensor(np.load(feat_path), device=model.device)
            try:
                start = int(splits[2]*model.fps)
                end = int(splits[3]*model.fps)
                feats = feats[start:end]
            except:
                pass
            feats = model.datamodule.normalize(feats)
            motion_feats.append(feats)
            motion_joints[i] = model.feats2joints(feats)
            motion_lengths[i] = feats.shape[0]
            
        texts.append(text.replace(
                '<Motion_Placeholder>', input_motion_holder_seq).replace(
                    '<Motion_Placeholder_s1>', input_motion_holder_seq).replace(
                        '<Motion_Placeholder_s2>', output_motion_holder_seq).replace(
                            '<Motion_Placeholder_Masked>', masked_holder_seq))

    return_dict = {
        'text': texts,
        'motion': motion_feats if len(motion_feats)>0 else None,
        'motion_joints': motion_joints,
        'motion_lengths': motion_lengths,
        # 'motion_token': motion_tokens_input,
        'motion_token_string': motion_token_string,
    }
    if len(motion_head) > 0:
        return_dict['motion_head'] = motion_head

    if len(motion_heading) > 0:
        return_dict['motion_heading'] = motion_heading

    if len(motion_tailing) > 0:
        return_dict['motion_tailing'] = motion_tailing

    return return_dict


def main():
    # parse options
    cfg = parse_args(phase="demo")  # parse config file
    cfg.FOLDER = cfg.TEST.FOLDER

    # create logger
    logger = create_logger(cfg, phase="test")

    task = cfg.DEMO.TASK
    text = None

    output_dir = Path(
        os.path.join(cfg.FOLDER, str(cfg.model.target.split('.')[-2]), str(cfg.NAME),
                     "samples_" + cfg.TIME))
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(OmegaConf.to_yaml(cfg))

    # set seed
    pl.seed_everything(cfg.SEED_VALUE)

    # gpu setting
    if cfg.ACCELERATOR == "gpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(
            str(x) for x in cfg.DEVICE)
        device = torch.device("cuda")

    # Dataset
    datamodule = build_data(cfg)
    logger.info("datasets module {} initialized".format("".join(
        cfg.DATASET.target.split('.')[-2])))

    # create model
    total_time = time.time()
    model = build_model(cfg, datamodule).eval()
    logger.info("model {} loaded".format(cfg.model.target))

    # loading state dict
    if cfg.TEST.CHECKPOINTS:
        logger.info("Loading checkpoints from {}".format(cfg.TEST.CHECKPOINTS))
        state_dict = torch.load(cfg.TEST.CHECKPOINTS,
                                map_location="cpu")["state_dict"]
        model.load_state_dict(state_dict, strict=False)
    else:
        logger.warning(
            "No checkpoints provided, using random initialized model")

    model.to(device)

    if cfg.DEMO.EXAMPLE:
        # Check txt file input
        # load txt
        return_dict = load_example_input(cfg.DEMO.EXAMPLE, task, model)
        text, in_joints = return_dict['text'], return_dict['motion_joints']
    else:
        assert False

    batch_size = 4
    from motGPT.data.utils import collate_tensors
    from motGPT.utils.render_utils import render_motion
    for b in tqdm(range(len(text) // batch_size + 1)):
        text_batch = text[b * batch_size:(b + 1) * batch_size]
        in_joints_batch = in_joints[b * batch_size:(b + 1) * batch_size]
        motion_lengths = return_dict["motion_lengths"][b * batch_size:(b + 1) * batch_size]
        batch = {
            "length":
            motion_lengths,
            "text":
            text_batch,
        }
        if return_dict['motion'] is not None:
            motion_feats = collate_tensors(return_dict['motion'][b*batch_size:(b+1)*batch_size]).to(device)
            batch["motion"] = motion_feats,
        
        print(task)
        if task in ['t2m', 't2t']:
            batch['motion_tokens_input'] = None
        else:
            motion_tokens_input, _ = model.lm.motion_feats_to_tokens(model.vae, motion_feats, motion_lengths, modes=task)
            batch['motion_tokens_input'] = motion_tokens_input
            
        outputs = model(batch, task=task)
        logger.info('Model forward finished! Start saving results...')
        if task in ['m2t', 't2t']:
            gen_texts = outputs['texts']
            for i in range(len(gen_texts)):
                idx = b * batch_size + i
                with open(os.path.join(output_dir, f'{idx}_out.txt'), 'w', encoding='utf-8') as f:
                    f.write(gen_texts[i])
                if task == 'm2t':
                    xyz = in_joints_batch[i][None, :motion_lengths[i]].cpu().detach()
                    # render_motion(xyz, xyz, output_dir=output_dir, fname=f'{idx}_in', fps=20)
                    np.save(os.path.join(output_dir, f'{idx}_in.npy'), xyz)
                else:
                    with open(os.path.join(output_dir, f'{idx}_in.txt'), 'w', encoding='utf-8') as f:
                        f.write(text_batch[i])

        else:
            joints = outputs["joints"]
            out_feats = outputs["feats"]
            lengths = outputs["length"]
            output_texts = outputs["texts"]
            for i in range(len(joints)):
                xyz = joints[i][:lengths[i]]
                xyz = xyz[None]
                out_feat = out_feats[i][:lengths[i]].detach().cpu().numpy()
                try:
                    xyz = xyz.detach().cpu().numpy()
                except:
                    xyz = xyz.detach().numpy()

                idx = b * batch_size + i
                if '<Motion_Placeholder>' in output_texts[i]:
                    # render_motion(xyz, xyz, output_dir=output_dir, fname=f'{idx}', fps=20)
                    np.save(os.path.join(output_dir, f'{idx}_out.npy'), xyz)
                    np.save(os.path.join(output_dir, f'{idx}_out_feats.npy'), out_feat)
                    # np.save(os.path.join(output_dir, f'{id}_in.npy'), xyz_in)

                with open(os.path.join(output_dir, f'{idx}_in.txt'), 'w') as f:
                    f.write(text_batch[i])
                    
                with open(os.path.join(output_dir, f'{idx}_out.txt'), 'w', encoding='utf-8') as f:
                    f.write(output_texts[i])
                    
                
                pose_vis = plot_3d.draw_to_batch(xyz, [text_batch[i]], [os.path.join(output_dir, f'{idx}_out.gif')])
                del pose_vis

    total_time = time.time() - total_time
    logger.info(
        f'Total time spent: {total_time:.2f} seconds (including model loading time and exporting time).'
    )
    logger.info(f"Testing done, the npy are saved to {output_dir}")


if __name__ == "__main__":
    with torch.no_grad():
        main()
