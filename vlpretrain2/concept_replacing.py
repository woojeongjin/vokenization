from numpy.lib.twodim_base import _trilu_indices_form_dispatcher
import spacy
import random
import copy
import pickle
# import tensorflow.compat.v1 as tf
class ConceptReplacer:
    def __init__(self, val, wiki=False):
        self.nlp = spacy.load('en_core_web_sm')
        self.nlp.pipeline = [("tagger", self.nlp.tagger), ("parser", self.nlp.parser)]
        if wiki:
            # /home/woojeong2/vok_pretraining/data/lxmert/train_cooccur_mscoco.pkl
            with open('/home/woojeong2/VL-BERT/data/en_corpus/train_cooccur_wiki.pkl', 'rb') as f:
                self.train_dict = pickle.load(f)
            with open('/home/woojeong2/VL-BERT/data/en_corpus/val_cooccur_wiki.pkl', 'rb') as f:
                self.val_dict = pickle.load(f)
        else:
            with open('/home/woojeong2/vok_pretraining/data/lxmert/train_cooccur_mscoco.pkl', 'rb') as f:
                self.train_dict = pickle.load(f)
            with open('/home/woojeong2/vok_pretraining/data/lxmert/val_cooccur_mscoco.pkl', 'rb') as f:
                self.val_dict = pickle.load(f)
        self.val = val
        if val:
            self.concept_dict = self.val_dict
        else:
            self.concept_dict = self.train_dict
        self.concept_dict_total = self.concept_dict[0].copy()
        self.concept_dict_total = self.concept_dict_total.update(self.concept_dict[1])
    #TODO : can generate concept shuffling ????
    def check_availability(self, sentence):
        def check_availability_sentence(x):
            # x = x.numpy().decode('utf-8')
            doc = self.nlp(str(x))
            V_concepts = []
            N_concepts = []
            original_tokens = []
            for token in doc:
                original_tokens.append(token.text_with_ws)
                if token.pos_.startswith('V') and token.is_alpha and not token.is_stop:
                    V_concepts.append(token.text_with_ws)
            for noun_chunk in doc.noun_chunks:
                root_noun = noun_chunk[-1]
                if root_noun.pos_ == "NOUN":
                    N_concepts.append(root_noun.text_with_ws)
            if len(N_concepts) >= 1 or len(V_concepts) >= 1:
                return True
            else:
                return False
        result = check_availability_sentence(sentence)
        # result = tf.py_function(check_availability_sentence,  [sentence], [tf.bool])[0]
        return result
    
    def concept_replace(self, prompt):
        doc = self.nlp(str(prompt))
        V_concepts = []
        V_concepts_lemma = [] 
        N_concepts = []
        N_concepts_lemma = []
        original_tokens = []
        for token in doc:
            original_tokens.append(token.text_with_ws)
            if token.pos_.startswith('V') and token.is_alpha and not token.is_stop:
                V_concepts.append(token.text_with_ws)
                V_concepts_lemma.append(token.lemma_)
        for noun_chunk in doc.noun_chunks:
            root_noun = noun_chunk[-1]
            if root_noun.pos_ == "NOUN":
                N_concepts.append(root_noun.text_with_ws)
                N_concepts_lemma.append(root_noun.lemma_)
        V = False
        N = False
        if len(V_concepts) >= 1 and len(N_concepts) >= 1:
            num = random.randint(0,1)
            if num == 0:
                V = True
            else:
                N = True
        elif len(V_concepts) >= 1:
            V = True
        elif len(N_concepts) >= 1:
            N = True

         
        shuffled_tokens = []
        if N is True and V is True:
            concepts = N_concepts + V_concepts
            concepts_lemma = N_concepts_lemma + V_concepts_lemma
            concept_dict = self.concept_dict_total
        elif N:
            concepts = N_concepts
            concepts_lemma = N_concepts_lemma
            concept_dict = self.concept_dict[0]
            
        elif V:
            concepts = V_concepts
            concepts_lemma = V_concepts_lemma
            concept_dict = self.concept_dict[1]
        # print('----------------')
        # print(len(concepts), N,V)
        idx = random.randint(0, len(concepts)-1)
        word = concepts[idx]
        word_lem = concepts_lemma[idx]
        assert len(concept_dict[word_lem]) != 0

        new_noun = random.choices(list(concept_dict[word_lem].keys()), weights=concept_dict[word_lem].values(), k=1)[0]
        shuffled_tok = []
        shuffled_id = []
        for i,tok in enumerate(original_tokens):
            if tok == word:
                shuffled_tokens.append(new_noun.strip())
                shuffled_tok.append(new_noun.lower().strip())
                shuffled_id.append(i)
            else:
                shuffled_tokens.append(tok.strip())
        
        
        # word = concepts
        # word_lem = concepts_lemma
        # for word_le in word_lem:
        #     assert len(concept_dict[word_le]) != 0
        # new_nouns = []
        # for word_le in word_lem:
        #     new_noun = random.choices(list(concept_dict[word_le].keys()), weights=concept_dict[word_le].values(), k=1)[0]
        #     new_nouns.append(new_noun)
        # shuffled_tok = []
        # shuffled_id = []

        # for i,tok in enumerate(original_tokens):
        #     if tok in word:
        #         idx = word.index(tok)
        #         shuffled_tokens.append(new_nouns[idx].strip())
        #         shuffled_tok.append(new_nouns[idx].lower().strip())
        #         shuffled_id.append(i)
        #     else:
        #         shuffled_tokens.append(tok.strip())

        
        assert len(shuffled_tokens) == len(original_tokens)
        result = ' '.join([token for token in shuffled_tokens])
        return result, shuffled_id, shuffled_tok

    def generate(self, prompt):
        return self.concept_replace(prompt)
        # negative_sampling = random.uniform(0,1) < 0.5
        # if negative_sampling:
        #     return self.cor_generate(prompt)
        # else:
        #     return self.c2s_generate(prompt)

