import random
import json
import numpy as np
from os.path import join as pjoin
from .dataset_t2m import Text2MotionDataset


class Text2MotionDatasetEvalV3(Text2MotionDataset):

    def __init__(
        self,
        data_root,
        split,
        mean,
        std,
        w_vectorizer,
        max_motion_length=196,
        min_motion_length=40,
        unit_length=4,
        fps=20,
        tmpFile=True,
        tiny=False,
        debug=False,
        stage='lm_pretrain',
        task_path=None,
        instruction_type='all',
        **kwargs,
    ):
        super().__init__(data_root, split, mean, std, max_motion_length,
                         min_motion_length, unit_length, fps, tmpFile, tiny,
                         debug, **kwargs)

        self.w_vectorizer = w_vectorizer

        # Load tasks for validation loss computation
        instruction_type_suffix = '' if (instruction_type == 'all') else '_' + instruction_type
        if task_path:
            instructions_path = task_path
        elif stage in ['lm_pretrain', 'lm_adaptor_pretrain', "lm_t2m"]:
            instructions_path = pjoin(data_root, f'template{instruction_type_suffix}_pretrain.json')
        elif stage in ['lm_instruct', "lm_rl", "lm_finetune"]:
            instructions_path = pjoin(data_root, f'template{instruction_type_suffix}_instructions.json')
        else:
            instructions_path = pjoin(data_root, f'template{instruction_type_suffix}_pretrain.json')

        self.instructions = json.load(open(instructions_path, 'r'))
        self.tasks = []
        for task in self.instructions.keys():
            for subtask in self.instructions[task].keys():
                self.tasks.append(self.instructions[task][subtask])


    def __getitem__(self, item):
        # Get text data
        idx = self.pointer + item
        fname = self.name_list[idx]
        data = self.data_dict[fname]
        motion, m_length, text_list = data["motion"], data["length"], data["text"]

        # all_captions = [
        #     ' '.join([token.split('/')[0] for token in text_dic['tokens']])
        #     for text_dic in text_list
        # ]
        all_captions = [
            text_dic['caption']
            for text_dic in text_list
        ]

        if len(all_captions) > 3:
            all_captions = all_captions[:3]
        elif len(all_captions) == 2:
            all_captions = all_captions + all_captions[0:1]
        elif len(all_captions) == 1:
            all_captions = all_captions * 3

        # Randomly select a caption
        text_data = random.choice(text_list)
        # text_data = text_list[0]
        caption, tokens = text_data["caption"], text_data["tokens"]
        # Text
        max_text_len = 20
        if len(tokens) < max_text_len:
            # pad with "unk"
            tokens = ["sos/OTHER"] + tokens + ["eos/OTHER"]
            sent_len = len(tokens)
            tokens = tokens + ["unk/OTHER"] * (max_text_len + 2 - sent_len)
        else:
            # crop
            tokens = tokens[:max_text_len]
            tokens = ["sos/OTHER"] + tokens + ["eos/OTHER"]
            sent_len = len(tokens)
        pos_one_hots = []
        word_embeddings = []
        for token in tokens:
            word_emb, pos_oh = self.w_vectorizer[token]
            pos_one_hots.append(pos_oh[None, :])
            word_embeddings.append(word_emb[None, :])
        pos_one_hots = np.concatenate(pos_one_hots, axis=0)
        word_embeddings = np.concatenate(word_embeddings, axis=0)

        # Random crop
        m_length = motion.shape[0]
        coin = np.random.choice([False, False, True])
        if coin:
            m_length = (m_length // self.unit_length - 1) * self.unit_length
        else:
            m_length = (m_length // self.unit_length) * self.unit_length

        # idx = 0
        idx = random.randint(0, len(motion) - m_length)
        motion = motion[idx:idx + m_length]

        # Z Normalization
        motion = (motion - self.mean) / self.std

        # Select a task (cycle through tasks based on item index)
        task_idx = item % len(self.tasks)
        tasks = self.tasks[task_idx]

        return caption, None, None, motion, m_length, word_embeddings, pos_one_hots, sent_len, "_".join(
            tokens), all_captions, tasks, fname
        # text, m_tokens, m_tokens_len, motion, length, word_embs, pos_ohot, text_len, tokens, all_captions ,tasks, fname
