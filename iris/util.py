import shlex
import numpy as np
import json

def single_or_list(x):
    if isinstance(x, list):
        return x
    if isinstance(x, tuple):
        return x
    else:
        return [x]

def is_data(result):
    if isinstance(result, np.ndarray):
        return True
    elif isinstance(result, dict):
        return True
    else:
        return False

def prettify_data(result):
    if isinstance(result, np.ndarray):
        return np.array_str(result)
    elif isinstance(result, dict):
        return json.dumps(result, indent=4, default=str)
    return result

# state machine util, conversation parsing

def get_start_message(messages): return messages[0]["text"]
def get_last_message(messages): return messages[-1]["text"]
def get_resolve_args_message(messages): return messages[0]["text"]

def verify_response(text):
    if text.lower() == "yes":
        return True
    else:
        return False

def verify_explain(text):
    if any([x in text.lower() for x in ["explain", "tell me more", "say more"]]):
        return True
    else:
        return False

def extract_number(text):
    for w in text.split():
        try:
            return True, int(w)
        except:
            pass
    return False, None

# is this word an argument?
def is_arg(s):
    if len(s)>2 and s[0] == "{" and s[-1] == "}": return True
    return False

# attempt to match query string to command and return mappings
def arg_match(query_string, command_string):#, types):
    maps = {}
    labels = []
    query_words, cmd_words = [shlex.split(x.lower()) for x in [query_string, command_string]]
    if len(query_words) != len(cmd_words): return False, {}
    for qw, cw in zip(query_words, cmd_words):
        if is_arg(cw):
            word_ = cw[1:-1]
            maps[word_] = qw #self.magic_type_convert(qw, types[word_])
        else:
            if qw != cw: return False, {}
    return True, maps
