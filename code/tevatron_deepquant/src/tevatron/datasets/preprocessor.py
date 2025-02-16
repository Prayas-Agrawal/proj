from . import deepquant_dataset as dq
from ..arguments import DataArguments
# from CQE import CQE
# def tokenize(self, batch_text, add_special_tokens=False):
#         assert type(batch_text) in [list, tuple], (type(batch_text))

#         tokens = [self.tok.tokenize(x, add_special_tokens=False) for x in batch_text]

#         if not add_special_tokens:
#             return tokens

#         prefix, suffix = [self.cls_token, self.Q_marker_token], [self.sep_token]
#         tokens = [prefix + lst + suffix + [self.mask_token] * (self.query_maxlen - (len(lst)+3)) for lst in tokens]

#         return tokens 

#     def encode(self, batch_text, add_special_tokens=False):
#         assert type(batch_text) in [list, tuple], (type(batch_text))

#         ids = self.tok(batch_text, add_special_tokens=False).to(DEVICE)['input_ids']

#         if not add_special_tokens:
#             return ids

#         prefix, suffix = [self.cls_token_id, self.Q_marker_token_id], [self.sep_token_id]
#         ids = [prefix + lst + suffix + [self.mask_token_id] * (self.query_maxlen - (len(lst)+3)) for lst in ids]
class UnitTensorizer:
    def __init__(self):
        self.units = {}  # Dictionary to store unit-to-number mapping
        self.next_available_num = 0  # Counter for the next available number

    def get(self, unit):
        """
        Get the numeric value for a unit. If the unit is not present, assign it the next available number.
        
        Args:
        - unit (str): The unit to check or add.
        
        Returns:
        - int: The numeric value for the unit.
        """
        if unit in self.units:
            return self.units[unit]
        
        # Assign the next available number to the new unit
        self.units[unit] = self.next_available_num
        self.next_available_num += 1
        return self.units[unit]
    

class TrainPreProcessor_deepquant:
    def __init__(self, tokenizer, query_max_length=32, text_max_length=256, separator=' '
                 , num_concepts_q=0, num_concepts_p=0, inject_concept_tokens=False, data_args: DataArguments = None):
        self.tokenizer = tokenizer
        self.query_max_length = query_max_length
        self.text_max_length = text_max_length
        self.separator = separator
        self.concept_string_q = ""
        self.concept_string_p = ""
        self.prefix = ["cls", "q/d"] if data_args.colbert_stanford_format else ["cls"]
        self.Q_marker_token, self.Q_marker_token_id = "[unused0]", tokenizer.convert_tokens_to_ids("[unused0]")
        self.P_marker_token, self.P_marker_token_id = "[unused1]", tokenizer.convert_tokens_to_ids("[unused1]")
        self.sep_token, self.sep_token_id = tokenizer.sep_token, tokenizer.sep_token_id
        self.start_offset_q = len(self.prefix)+num_concepts_q
        self.start_offset_p = len(self.prefix)+num_concepts_p
        self.all_units = UnitTensorizer()
        # self.parser = CQE.CQE(overload=True)
        self.parser = None
        self.data_args = data_args
        if(inject_concept_tokens):
            self.concept_string_q = "".join(["<concept>" for _ in range(num_concepts_q)]) if num_concepts_q else ""
            self.concept_string_p = "".join(["<concept>" for _ in range(num_concepts_p)]) if num_concepts_p else ""
            
    def process(self,prefix2, meta, text, start_offset, max_length, max_masks, prefix, suffix, query, id=None):
        proc_text, numbers = dq.process_text_deepquant(prefix2, meta, text, self.tokenizer, 
                                                       start_offset, max_length, max_masks, 
                                                       self.data_args, query=query, parser=self.parser, id=id, all_units=self.all_units)
        proc_text = prefix + " " + proc_text + " " + suffix if self.data_args.colbert_stanford_format else proc_text
        return proc_text, numbers
    
    def process_psgs(self, query, passages):
        docs = []
        numbers_docs = []
        # print("qry_text", qry_text, flush=True)
        for pos in passages:
            text_prefix = ""
            text = pos['text']
            # docid
            docid = pos["docid"]
            if ('title' in pos and len(pos['title']) > 0):
                text_prefix = ""
                # text = self.concept_string_p + pos['title'] + self.separator + pos['text']
                text = self.concept_string_p + pos['text']
            text, numbers = self.process(text_prefix, pos["meta"], text, self.start_offset_p, self.text_max_length, 
                                            self.data_args.deepquant_p_maxnums, 
                                            self.P_marker_token, self.sep_token, query=query, id=docid)
            
            if(numbers is not None):
                numbers_docs.append(numbers)
            docs.append(self.tokenizer.encode(text,
                                                add_special_tokens=False,
                                                max_length=self.text_max_length,
                                                truncation=True))
        return docs, numbers_docs

    def __call__(self, example):
        try:
            qry_prefix = self.concept_string_q
            qry_text = qry_prefix + example['query']
            qid = example["query_id"]
            qry, numbers_qry = self.process(qry_prefix, example["meta"], qry_text, self.start_offset_q, self.query_max_length, 
                                            self.data_args.deepquant_q_maxnums, self.Q_marker_token, 
                                            self.sep_token, query=qry_text, id=qid)
            
            query = self.tokenizer.encode(qry,
                                        add_special_tokens=False,
                                        max_length=self.query_max_length,
                                        truncation=True)
            
            positives, numbers_positives = self.process_psgs(qry_text, example["positive_passages"])
            negatives, numbers_negatives = self.process_psgs(qry_text, example["negative_passages"])
            
            # return {'query': query, 'positives': positives, 'negatives': negatives}
            return {'query': query, 'positives': positives, 'negatives': negatives, 
                    "numbers_qry": numbers_qry, "numbers_pos": numbers_positives, "numbers_neg": numbers_negatives}
        except:
            # print("Failed at", example)
            raise "FAILED"
        

class TrainPreProcessor:
    def __init__(self, tokenizer, query_max_length=32, text_max_length=256, separator=' '
                 , num_concepts_q=0, num_concepts_p=0, inject_concept_tokens=False, data_args: DataArguments = None):
        self.tokenizer = tokenizer
        self.query_max_length = query_max_length
        self.text_max_length = text_max_length
        self.separator = separator
        self.concept_string_q = ""
        self.concept_string_p = ""
        self.data_args = data_args
        if(inject_concept_tokens):
            self.concept_string_q = "".join(["<concept>" for _ in range(num_concepts_q)]) if num_concepts_q else ""
            self.concept_string_p = "".join(["<concept>" for _ in range(num_concepts_p)]) if num_concepts_p else ""
            
        
    def __call__(self, example):
        # try:
        qry = self.concept_string_q + example['query']
        
        query = self.tokenizer.encode(qry,
                                    add_special_tokens=False,
                                    max_length=self.query_max_length,
                                    truncation=True)
        positives = []
        numbers_positives = []
        for pos in example['positive_passages']:
            text = self.concept_string_p + pos['title'] + self.separator + pos['text'] if 'title' in pos else pos['text']
            positives.append(self.tokenizer.encode(text,
                                                add_special_tokens=False,
                                                max_length=self.text_max_length,
                                                truncation=True))
        negatives = []
        for neg in example['negative_passages']:
            text = self.concept_string_p + neg['title'] + self.separator + neg['text'] if 'title' in neg else neg['text']
            negatives.append(self.tokenizer.encode(text,
                                                add_special_tokens=False,
                                                max_length=self.text_max_length,
                                                truncation=True))
        return {'query': query, 'positives': positives, 'negatives': negatives}

class QueryPreProcessor:
    def __init__(self, tokenizer, query_max_length=32):
        self.tokenizer = tokenizer
        self.query_max_length = query_max_length

    def __call__(self, example):
        query_id = example['query_id']
        query = self.tokenizer.encode(example['query'],
                                      add_special_tokens=False,
                                      max_length=self.query_max_length,
                                      truncation=True)
        return {'text_id': query_id, 'text': query}


class CorpusPreProcessor:
    def __init__(self, tokenizer, text_max_length=256, separator=' '):
        self.tokenizer = tokenizer
        self.text_max_length = text_max_length
        self.separator = separator

    def __call__(self, example):
        docid = example['docid']
        text = example['title'] + self.separator + example['text'] if 'title' in example else example['text']
        text = self.tokenizer.encode(text,
                                     add_special_tokens=False,
                                     max_length=self.text_max_length,
                                     truncation=True)
        return {'text_id': docid, 'text': text}