import numpy as np
import torch
import pickle
from tevatron.arguments import DataArguments
import pandas as pd
import copy

def is_whitespace(c):
    if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
        return True
    return False

ADDED = dict()

def replace_multiple(text, idxs,):
    newtext = text + ""
    offset = 0 
    newIdxs = copy.deepcopy(idxs)
    def rep(s, idxs, offset):
        start, end = idxs
        adjusted_start = start + offset
        adjusted_end = end + offset
        
        s = s[:adjusted_start] + "<num>" + s[adjusted_end:]
        
        offset += len("<num>") - (adjusted_end - adjusted_start)
        newIdx = (adjusted_start, adjusted_start+len("<num>"))
        return s, offset, newIdx
    for i, num in enumerate(idxs):
        for j, _num in enumerate(num):
            newtext, offset, newIdx = rep(newtext, _num, offset)
            # newIdxs[i][j] = (_num[0], _num[0]+1) FIXME: Change this ?
            newIdxs[i][j] = newIdx
    return newtext, newIdxs

def adjust_indices_and_replace(text, idxs1, idxs_op, replace_string):
    idxs1 = [list(pair) for pair in idxs1]
    a2_updated = []

    offset = 0
    for i, (start, end) in enumerate(idxs_op):
        replace_len = len(replace_string[i])
        
        original_len = end - start
        text = text[:start + offset] + replace_string[i] + text[end + offset:]
        diff = replace_len - original_len

        for idx, (a1_start, a1_end) in enumerate(idxs1):
            if a1_start >= end + offset:
                # Entire range is after the replacement, just shift by diff
                idxs1[idx][0] += diff
                idxs1[idx][1] += diff
            elif a1_end <= start + offset:
                # Entire range is before the replacement, no change needed
                continue
            else:
                # Range overlaps with the replacement; adjust accordingly
                if a1_start >= start + offset:
                    idxs1[idx][0] = max(a1_start + diff, start + offset)
                if a1_end >= start + offset:
                    idxs1[idx][1] = max(min(a1_end + diff, end + offset + diff), idxs1[idx][0])

        new_start = start + offset
        new_end = new_start + replace_len
        a2_updated.append((new_start, new_end))

        offset += diff

    return text, [tuple(pair) for pair in idxs1], a2_updated
def pickle_load(path):

    try:
        f = None
        if(path is None or len(path.strip()) == 0): return None
        with open(path, "rb") as fp:
            f = pickle.load(fp)
        return f
    except:
        return None
    
ALL_UNITS = None
def process_text_deepquant(prefix, meta, text, tokenizer, start_offset=1, max_length = 16, 
                              max_masks = 6, data_args:DataArguments = None, query=None,id=None, whole_obj=None, all_units=None, **kwargs):
    global ALL_UNITS
    if(id is None): raise "Give ids"
    if(ALL_UNITS is None):
        ALL_UNITS = pickle_load(data_args.all_units)
        

    def overlaps(token_range, char_range):
        token_start, token_end = token_range
        if(len(char_range) == 0 or char_range is None): return False
        char_start, char_end = char_range[0]
        return token_start < char_end and token_end > char_start
    
    def get_op(meta):
        key = None
        if "condition" in meta:
            key = "condition"
        if "op" in meta:
            if key is not None: raise "Found both op and condition keys"
            key = "op"
        if key is None:
            return torch.tensor([0,0,0]).numpy()
        return torch.nn.functional.one_hot(torch.tensor([">", "<", "="].index(meta[key])), 3).numpy()

    number_masks = []
    numbers = meta["values"][:max_masks]
    # numbers = [max(min(num, 1e20), -1e20) for num in numbers]
    # units = [all_units.get(u) if all_units is not None else -1  for u in meta["units"]][:max_masks]
    units = [ALL_UNITS[u] if ALL_UNITS is not None and u in ALL_UNITS else -1 for u in meta["units"]][:max_masks]
    # units = [ -1 for u in meta["units"]][:max_masks]
    number_len = [0 for i in range(max_masks)]
    
    # FIXME: need to change unit idxs as well
    if(data_args.num_as_token and not data_args.inject_units):
        text, meta['value_char_indices'] = replace_multiple(text, meta['value_char_indices'])
    elif(data_args.num_as_token and data_args.inject_units):
        flat_num = [idx_list[0] for idx_list in meta["value_char_indices"]]
        flat_unit = [idx for idx_list in meta["unit_char_indices"] for idx in idx_list ]
        text, flat_unit, flat_num = adjust_indices_and_replace(text, flat_unit, flat_num,
                                                        ["<num>" for _ in range(len(flat_num))])
        
    if(data_args.inject_units):
        # text, meta['unit_char_indices'] = replace_unit(num_idxs=meta['value_char_indices'], text=text, 
                                                        # unit_idxs=meta['unit_char_indices'], units=meta["units"])
        #Clear units text
        text, flat_num, flat_unit = adjust_indices_and_replace(text, flat_num, flat_unit,
                                                        ['' for _ in range(len(flat_unit))])
        
        flat_unit = [(idx[1], idx[1]) for idx in flat_num]
        assert len(flat_num) == len(meta["units"])
        
        text, flat_num, flat_unit = adjust_indices_and_replace(text, flat_num, flat_unit,
                                                        [u for u in meta['units']])
        meta['value_char_indices'] = [[idx] for idx in flat_num]
        meta['unit_char_indices'] = [[idx] for idx in flat_unit]
        
        # for i, num in enumerate(meta['value_char_indices']):
        #     for j, _num in enumerate(num):
        #         text = text[:_num[0]] + "<num>" + text[_num[1]:]
        #         meta['value_char_indices'][i][j] = (_num[0], _num[0]+1)
    
    encoding = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False, max_length=max_length-start_offset)
    token_offsets = encoding['offset_mapping']
    
    for i, num in enumerate(numbers):
        number_len[i] = len(str(num))
    
    for char_range in (meta['value_char_indices']):
        mask = [0] * (max_length)
        for i, token_range in enumerate(token_offsets):
            if overlaps(token_range, char_range):
                mask[i + start_offset] = 1
        number_masks.append(np.array(mask))

    unit_masks = []
    for char_range in (meta['unit_char_indices']):
        mask = [0] * (max_length)
        for i, token_range in enumerate(token_offsets):
            if overlaps(token_range, char_range):
                mask[i + start_offset] = 1
        unit_masks.append(np.array(mask))
        
    number_masks = number_masks[:max_masks]
    unit_masks = unit_masks[:max_masks]
     
        
    if(len(number_masks) < max_masks):
        diff = max_masks - len(number_masks)
        numbers.extend([0 for i in range(diff)])
        units.extend([-1 for i in range(diff)])
        
        number_masks.extend([np.array([0 for _ in range(max_length)]) for _ in range(diff)])
        
    if(len(unit_masks) < max_masks):
        diff = max_masks - len(unit_masks)
        unit_masks.extend([np.array([0 for _ in range(max_length)]) for _ in range(diff)])
    
    ret = {
        'numbers': torch.tensor(numbers, dtype=torch.float64),
        'units': torch.tensor(units),
        'number_mask': torch.tensor(number_masks, dtype=torch.int8),
        'unit_mask': torch.tensor(unit_masks, dtype=torch.int8),
        'number_len': torch.tensor(number_len, dtype=torch.int8),
        'id': torch.tensor(int(id)),
        'op': get_op(meta),
        'score': torch.tensor(float(whole_obj.get("score"))) if whole_obj is not None else torch.tensor(0.0)
    }
    
    
    return text, ret
