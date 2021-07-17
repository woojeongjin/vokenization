import spacy
import random
import copy
# import tensorflow.compat.v1 as tf
class ConceptGenerator:
    def __init__(self, randomnum=False, k =0):
        self.nlp = spacy.load('en_core_web_sm')
        self.nlp.pipeline = [("tagger", self.nlp.tagger), ("parser", self.nlp.parser)]
        self.randomnum = randomnum
        self.k = k
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
            if len(N_concepts) >= 2 or len(V_concepts) >= 2:
                if len(set(N_concepts)) == 1 or len(set(V_concepts)) == 1:
                    return False
                else:
                    return True
            else:
                return False
        result = check_availability_sentence(sentence)
        # result = tf.py_function(check_availability_sentence,  [sentence], [tf.bool])[0]
        return result
    def cor_generate(self, prompt):
        doc = self.nlp(str(prompt))
        V_concepts = []
        V_concepts_id = []
        N_concepts = []
        N_concepts_id = []
        original_tokens = []
        for i, token in enumerate(doc):
            original_tokens.append(token.text_with_ws)
            if token.pos_.startswith('V') and token.is_alpha and not token.is_stop:
                V_concepts.append(token.text_with_ws)
                V_concepts_id.append(i)
        for noun_chunk in doc.noun_chunks:
            root_noun = noun_chunk[-1]
            if root_noun.pos_ == "NOUN":
                N_concepts.append(root_noun.text_with_ws)
                N_concepts_id.append(noun_chunk.end-1)

        if len(N_concepts) >= 2:
            if self.randomnum:
                if self.k == 0:
                    k = random.randint(2,len(N_concepts))
                else:
                    k = 2
                random_ids = sorted(random.sample(range(len(N_concepts_id)), k))
                N_concepts_id_tem = [N_concepts_id[i] for i in random_ids]
                N_concepts_tem = [N_concepts[i] for i in random_ids]
                while len(set(N_concepts_tem)) == 1:
                    if self.k == 0:
                        k = random.randint(2,len(N_concepts))
                    else:
                        k = 2
                    random_ids = sorted(random.sample(range(len(N_concepts_id)), k))
                    N_concepts_id_tem = [N_concepts_id[i] for i in random_ids]
                    N_concepts_tem = [N_concepts[i] for i in random_ids]
                
                N_concepts = N_concepts_tem
                N_concepts_id = N_concepts_id_tem

            for i in range(len(N_concepts)):
                assert doc[N_concepts_id[i]].text_with_ws == N_concepts[i]
            
            previous_id = copy.deepcopy(N_concepts_id)
            if len(N_concepts_id) == 2:
                # tem = N_concepts.pop(0)
                # N_concepts.append(tem)
                tem = N_concepts_id.pop(0)
                N_concepts_id.append(tem)
            else:
                while previous_id == N_concepts_id:
                    random.shuffle(N_concepts_id)


        if len(V_concepts) >= 2:
            if self.randomnum:
                if self.k == 0:
                    k = random.randint(2,len(V_concepts))
                else:
                    k = 2
                random_ids = sorted(random.sample(range(len(V_concepts_id)), k))
                V_concepts_id_tem = [V_concepts_id[i] for i in random_ids]
                V_concepts_tem = [V_concepts[i] for i in random_ids]
                while len(set(V_concepts_tem)) == 1:
                    if self.k == 0:
                        k = random.randint(2,len(V_concepts))
                    else:
                        k = 2
                    random_ids = sorted(random.sample(range(len(V_concepts_id)), k))
                    V_concepts_id_tem = [V_concepts_id[i] for i in random_ids]
                    V_concepts_tem = [V_concepts[i] for i in random_ids]
                
                V_concepts = V_concepts_tem
                V_concepts_id = V_concepts_id_tem

            for i in range(len(V_concepts)):
                assert doc[V_concepts_id[i]].text_with_ws == V_concepts[i]
            
            previous_id = copy.deepcopy(V_concepts_id)
            if len(V_concepts_id) == 2:
                # tem = V_concepts.pop(0)
                # V_concepts.append(tem)
                tem = V_concepts_id.pop(0)
                V_concepts_id.append(tem)
            else:
                while previous_id == V_concepts_id:
                    random.shuffle(V_concepts_id)

        shuffled_tokens = []
        N_concepts_index = 0
        V_concepts_index = 0
        shuffled_tok = []
        shuffled_id = []

        for i,tok in enumerate(original_tokens):
            if i in V_concepts_id and V_concepts_index < len(V_concepts_id):
                shuffled_tokens.append(original_tokens[V_concepts_id[V_concepts_index]].strip())
                shuffled_tok.append(original_tokens[V_concepts_id[V_concepts_index]].lower().strip())
                V_concepts_index += 1
                shuffled_id.append(i)
            elif i in N_concepts_id and N_concepts_index < len(N_concepts_id):
                shuffled_tokens.append(original_tokens[N_concepts_id[N_concepts_index]].strip())
                shuffled_tok.append(original_tokens[N_concepts_id[N_concepts_index]].lower().strip())
                N_concepts_index += 1
                shuffled_id.append(i)
            else:
                shuffled_tokens.append(tok.strip())   
        # for i,tok in enumerate(original_tokens):
        #     if tok in V_concepts and V_concepts_index < len(V_concepts):
        #         shuffled_tokens.append(V_concepts[V_concepts_index].strip())
        #         V_concepts_index += 1
        #         shuffled_tok.append(tok.lower().strip())
        #         shuffled_id.append(i)
        #     elif tok in N_concepts and N_concepts_index < len(N_concepts):
        #         shuffled_tokens.append(N_concepts[N_concepts_index].strip())
        #         N_concepts_index += 1
        #         shuffled_tok.append(tok.lower().strip())
        #         shuffled_id.append(i)
        #     else:
        #         shuffled_tokens.append(tok.strip())
        assert len(shuffled_tokens) == len(original_tokens)
        result = ' '.join([token for token in shuffled_tokens])
        return result, shuffled_id, shuffled_tok
    def c2s_generate(self, prompt):
        doc = self.nlp(str(prompt))
        matched_concepts = []
        for token in doc:
            if (token.pos_.startswith('V') or token.pos_.startswith('PROP')) and token.is_alpha and not token.is_stop:
                matched_concepts.append(token.lemma_)
        for noun_chunk in doc.noun_chunks:
            root_noun = noun_chunk[-1]
            if root_noun.pos_ == "NOUN":
                matched_concepts.append(root_noun.lemma_)
        result = " ".join([token for token in matched_concepts])
        return result
    def generate(self, prompt):
        return self.cor_generate(prompt)
        # negative_sampling = random.uniform(0,1) < 0.5
        # if negative_sampling:
        #     return self.cor_generate(prompt)
        # else:
        #     return self.c2s_generate(prompt)


# import spacy
# import random
# import copy
# import tensorflow.compat.v1 as tf
# class ConceptGenerator:
#     def __init__(self):
#         self.nlp = spacy.load('en_core_web_sm')
#         self.nlp.pipeline = [("tagger", self.nlp.tagger), ("parser", self.nlp.parser)]
#     #TODO : can generate concept shuffling ????
    
#     def check_availability(self, sentence):
#         def check_availability_sentence(x):
#             x = x.numpy().decode('utf-8')
#             doc = self.nlp(str(x))
#             V_concepts = []
#             N_concepts = []
#             original_tokens = []
#             for token in doc:
#                 original_tokens.append(token.text_with_ws)
#                 if token.pos_.startswith('V') and token.is_alpha and not token.is_stop:
#                     V_concepts.append(token.text_with_ws)
#             for noun_chunk in doc.noun_chunks:
#                 root_noun = noun_chunk[-1]
#                 if root_noun.pos_ == "NOUN":
#                     N_concepts.append(root_noun.text_with_ws)
#             if len(N_concepts) >= 2 or len(V_concepts) >= 2:
#                 if len(set(N_concepts)) == 1 or len(set(V_concepts)) == 1:
#                     return False
#                 else:
#                     return True
#             else:
#                 return False
#         result = tf.py_function(check_availability_sentence, [sentence], [tf.bool])[0]
#         return result
    
#     def cor_generate(self, prompt):
#         doc = self.nlp(str(prompt))
#         V_concepts = []
#         N_concepts = []
#         original_tokens = []
#         for token in doc:
#             original_tokens.append(token.text_with_ws)
#             if token.pos_.startswith('V') and token.is_alpha and not token.is_stop:
#                 V_concepts.append(token.text_with_ws)
#         for noun_chunk in doc.noun_chunks:
#             root_noun = noun_chunk[-1]
#             if root_noun.pos_ == "NOUN":
#                 N_concepts.append(root_noun.text_with_ws)
#         if len(N_concepts) >= 2:
#             previous = copy.deepcopy(N_concepts)
#             while previous == N_concepts:
#                 random.shuffle(N_concepts)
#         if len(V_concepts) >= 2:
#             previous = copy.deepcopy(V_concepts)
#             while previous == V_concepts:
#                 random.shuffle(V_concepts)
#         shuffled_tokens = []
#         N_concepts_index = 0
#         V_concepts_index = 0
#         for tok in original_tokens:
#             if tok in V_concepts and V_concepts_index < len(V_concepts):
#                 shuffled_tokens.append(V_concepts[V_concepts_index])
#                 V_concepts_index += 1
#             elif tok in N_concepts and N_concepts_index < len(N_concepts):
#                 shuffled_tokens.append(N_concepts[N_concepts_index])
#                 N_concepts_index += 1
#             else:
#                 shuffled_tokens.append(tok)
#         assert len(shuffled_tokens) == len(original_tokens)
#         result = ''.join([token for token in shuffled_tokens])
#         return result
    
#     def c2s_generate(self, prompt):
#         doc = self.nlp(str(prompt))
#         matched_concepts = []
#         for token in doc:
#             if (token.pos_.startswith('V') or token.pos_.startswith('PROP')) and token.is_alpha and not token.is_stop:
#                 matched_concepts.append(token.lemma_)
#         for noun_chunk in doc.noun_chunks:
#             root_noun = noun_chunk[-1]
#             if root_noun.pos_ == "NOUN":
#                 matched_concepts.append(root_noun.lemma_)
#         result = " ".join([token for token in matched_concepts])
#         return result
    
#     def generate(self, prompt):

#         return self.cor_generate(prompt)
        # negative_sampling = random.uniform(0,1) < 0.5
        # if negative_sampling:
        #     return self.cor_generate(prompt)
        # else:
        #     return self.c2s_generate(prompt)



# generator = ConceptGenerator()
# if generator.check_availability(sent):
#     generated_sentence = generator.generate(sent)


