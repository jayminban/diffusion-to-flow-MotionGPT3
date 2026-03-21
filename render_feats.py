import os
import numpy as np
from mGPTv3.utils.render_utils import render_motion

fps=20
path = r'D:\0-0\Desktop\MoTGPT\MoTionGPT3\assets\video\t2m'
for fn in os.listdir(os.path.join(path, 'jnts')):
    if not fn.endswith('.npy'): continue
    feats = np.load(os.path.join(path, 'jnts', fn))
    # render_motion(feats, feats.cpu().numpy(), output_dir=path, fname=fn.split('.')[0]+'_fast',method='fast', fps=fps)
    render_motion(feats, feats, output_dir=path, fname=fn.split('.')[0],method='slow', fps=fps)