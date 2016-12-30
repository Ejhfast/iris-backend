import shlex

# state machine util, conversation parsing

def get_start_message(messages): return messages[0]["text"]
def get_classification_message(messages): return messages[-1]["text"]
def get_resolve_args_message(messages): return messages[0]["text"]

def verify_response(text):
    if text.lower() == "yes":
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

# this is for parsing args out of iris conversation
def parse_message_args(messages):
    fail_indexes = [i for i,x in enumerate(messages) if x["origin"] == "iris" and x["state"] == "RESOLVE_ARGS" ]
    args = {}
    for i in fail_indexes:
        iris_ask = messages[i]["text"]
        var = messages[i]["arg"] #iris_ask.split()[-1][:-1]
        args[var] = messages[i+1]["text"]
    return args

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
