# coding=utf-8
import json
from pathlib import Path
import random

from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader
from vlpretrain.concept_swapping import ConceptGenerator
from vlpretrain.concept_replacing import ConceptReplacer

from PIL import Image
Image.MAX_IMAGE_PIXELS = None

TINY_IMG_NUM = 1000
FAST_IMG_NUM = 10000

lxrt_imgsplits = {
    'mscoco_train',
    'mscoco_nominival',
    'vgnococo',
    'mscoco_minival',
}
lxrt_langsplits = {
    'mscoco', 'vg', 'vqa', 'gqa', 'visual7w'
}
cc_imgsplits = {
    'cc_train': 'training.tsv',
    'cc_valid': 'validation.tsv',
}
cc_langsplits = {
    'cc',
}

CC_ROOT = 'data/cc'
COCO_ROOT = 'data/mscoco'
VG_ROOT = '/ssd-playpen/data/vg'
LXRT_ROOT = 'data/lxmert'


def make_uid(img_id, source, sent_id):
    """
    see the descriptions in function 'make_datum'
    """
    return "%s:%s:%s" % (img_id, source, sent_id)


def get_img_path(source, img_id):
    if source == 'cc':
        split_tag, _ = img_id.split('_')
        return "%s/images/%s/%s" % (CC_ROOT, split_tag, img_id)
    elif 'COCO' in img_id:
        _, split_tag, _ = img_id.split('_')
        return "%s/images/%s/%s" % (COCO_ROOT, split_tag, img_id + '.jpg')
    else:   # VG images
        return "%s/images/%s.jpg" % (VG_ROOT, img_id)


def make_datum(source: str, img_id: str, sent_id: int, sent: str):
    """
    Create a datum from the provided infos.
    :param source: the dataset of the particular sentence.
    :param img_id: id of the image
    :param sent_id: id of the sentence (of the image)
    :param sent: the sentence
    :return: a dict of datum
    """
    uid = make_uid(img_id, source, sent_id)
    img_path = get_img_path(source, img_id)
    return {
        'uid': uid,
        'img_id': img_id,
        'img_path': img_path,
        'sent': sent,
    }


class ImgSentDataset:
    def __init__(self, img_splits: str, lang_splits: str, tiny=False, fast=False, val=False):
        """
        :param split: train, valid, test
        :param sources: The data sources to be loaded, separated by comma.
                       from: mscoco, cc, vg, vqa, gqa, visual7w
                             'vg' stands for visual genome captions
                             'cc' stands for conceptual captions.
                       example: 'mscoco, vg'
        """
        self.img_splits = [img_split.lower().strip() for img_split in img_splits.split(',')]
        self.lang_splits = [lang_split.lower().strip() for lang_split in lang_splits.split(',')]
        self.data = []
        self.generator = ConceptGenerator()

        data_size = 30000
        # data_size = 100000
        debug_imgs = -1
        if tiny:
            debug_imgs = TINY_IMG_NUM
        elif fast:
            debug_imgs = FAST_IMG_NUM

        # Loading LXRT data (i.e., COCO Cap, VQA, GQA, VG Cap, VG QA (visual7w))
        lxrt_data = []
        lxrt_path = Path(LXRT_ROOT)

        cc = False
        # cc
        if cc:
            with open('/home/woojeong2/vok_pretraining/data/conceptual-captions/train.json','r') as f:
                cc_data = []
                for line in f.readlines():
                    cc_data.append(json.loads(line))
            ####
        wiki = True
        if wiki:
            corpus = []
            with open('/home/woojeong2/VL-BERT/data/en_corpus/wiki.doc', 'r', encoding='utf-8') as f:
                corpus.extend([l.strip('\n').strip('\r').strip('\n') for l in f.readlines()])

            corpus = [l.strip() for l in corpus if l.strip() != '']
        for img_split in self.img_splits:
            if img_split in lxrt_imgsplits:
                fname = img_split + ".json"
                if debug_imgs > 0 and fname != 'mscoco_nominival.json' \
                        and fname != 'mscoco_minival.json':  # Only load nominival when debugging
                    continue
                lxrt_data.extend(json.load((lxrt_path / fname).open()))

        for i, lxrt_datum in enumerate(lxrt_data):
            if i % 1000 == 0:
                print(i, len(lxrt_data))
            img_id = lxrt_datum['img_id']
            for lang_split in self.lang_splits:
                if lang_split in lxrt_datum['sentf']:
                    # for cc
                    if cc:
                        sent = " ".join(cc_data[i]['caption'])
                        if self.generator.check_availability(sent):
                            self.data.append(make_datum(lang_split, img_id, 0, sent))
                    ##
                    elif wiki:
                        if val:
                            sent = corpus[data_size+i]
                        else:
                            sent = corpus[i]
                        if self.generator.check_availability(sent):
                            self.data.append(make_datum(lang_split, img_id, 0, sent))
                    else:
                        sents = lxrt_datum['sentf'][lang_split] # mscoco sentence
                        for j, sent in enumerate(sents):
                            if self.generator.check_availability(sent):
                                self.data.append(make_datum(lang_split, img_id, j, sent))
                            # if debug_imgs > 0:  # Only load one sentence if debugging
                            #     break
            if i+1 == debug_imgs:             # Load top #debug_imgs images
                break
            
            if i+1 == data_size:
                break

        # Loading Conceptual Caption (CC) data
        for img_split in self.img_splits:
            if img_split in cc_imgsplits:
                cc_path = Path(CC_ROOT)
                for fname in cc_imgsplits[img_split]:
                    for i, line in enumerate((cc_path / fname).open()):
                        sent, img_id = line.split('\t')
                        self.data.append(make_datum('cc', img_id.strip(), 0, sent))
                        if i+1 == debug_imgs:
                            break

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]

    # def shuffle(self):
    #     random.seed(9595)
    #     random.shuffle(self.data)


class ImgSentTorchDataset(Dataset):
    def __init__(self,
                 dataset: ImgSentDataset,
                 img_transform,
                 tokenizer,
                 sent_len: int,
                 val=False,
                 perturb="swap"):
        super().__init__()
        self.raw_dataset = dataset
        self.img_transform = img_transform
        self.tokenizer = tokenizer
        self.sent_len = sent_len
        if perturb == 'swap':
            self.generator = ConceptGenerator()
        else:
            self.replacer = ConceptReplacer(val)
        self.perturb = perturb

    def __len__(self):
        return len(self.raw_dataset)

    def __getitem__(self, item: int):
        datum = self.raw_dataset[item]

        uid = datum['uid']
        img_id = datum['img_id']
        img_path = datum['img_path']
        sent = datum['sent']

        # Step 1: Load and pre-process the image
        try:
            pil_img = default_loader(img_path)
        except Exception as e:
            print(e)
            print(img_path)
            return self.__getitem__((item + 95) % self.__len__())
        tensor_img = self.img_transform(pil_img)

        # Step 2: Tokenization (to integers) and Padding
        encoded_sent = self.tokenizer.encode_plus(
            sent,
            add_special_tokens=True,
            max_length=self.sent_len,
            truncation=True,
            # pad_to_max_length=True,
            padding='max_length',
            return_tensors='pt'     # Return PyTorch (pt) tensors
        )
        input_ids = encoded_sent['input_ids'].squeeze()
        attention_mask = encoded_sent['attention_mask'].squeeze()
        # print('sent', sent)
        # print('input_ids', input_ids)
        # print('attention_mask', attention_mask)
        
        
        # Negative examples
        
        # if self.generator.check_availability(sent):
        if self.perturb == 'swap':
            generated_sent = self.generator.generate(sent)
        else:
            generated_sent = self.replacer.generate(sent)

        # print(uid)


        neg_encoded_sent = self.tokenizer.encode_plus(
            generated_sent,
            add_special_tokens=True,
            max_length=self.sent_len,
            truncation=True,
            # pad_to_max_length=True,
            padding='max_length',
            return_tensors='pt'     # Return PyTorch (pt) tensors
        )
        neg_input_ids = neg_encoded_sent['input_ids'].squeeze()
        neg_attention_mask = neg_encoded_sent['attention_mask'].squeeze()


        return uid, (input_ids, attention_mask, ), (neg_input_ids, neg_attention_mask, ), (tensor_img, ), sent, generated_sent
        # return uid, (input_ids, attention_mask, ), (neg_input_ids, neg_attention_mask, ), (tensor_img, )
