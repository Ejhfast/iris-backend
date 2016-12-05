
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
