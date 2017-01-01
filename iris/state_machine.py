from . import util
from .core import IRIS
from . import iris_types as t

def process_succ_failure(iris, cls_idx, s_args, arg_names, query):
    succs = [x[0] for x in s_args]
    args = [x[1] for x in s_args]
    values = [x[2] for x in s_args]
    triples = list(zip(arg_names,args,succs))
    for_ex = list(zip(arg_names,values,succs))
    if all(succs):
        learn, lcmd = iris.learn_from_example(cls_idx, query, for_ex)
        result = iris.class_functions[cls_idx]["function"](*args)
        iris.train_model()
        if isinstance(result, t.IrisImage):
            result = {"type":"image", "value":result.value}
        # elif isinstance(result, IrisValue):
        #     # FIXME: support this better, mutliple iris values
        #     if isinstance(result, IrisValues):
        #         name = ", ".join(result.names)
        #     else:
        #         name = result.name
        #     result = "I stored the result in \"{}\"".format(name)
        else:
            result = iris.class2format[cls_idx](result)
        if learn:
            result = ["(I learned how to \"{}\")".format(lcmd),result]
        else:
            result = [result]
        return result
    else:
        # kind of obnoxious that I am doing string construction here...
        assump = lambda x,y: "I think {} is \"{}\"".format(x,y)
        problem = lambda x,y: "For {}, {}".format(x,y)
        arg_assumptions = [assump(x[0],x[1]) for x in triples if x[2] == True]
        arg_problems = [problem(x[0],x[1]) for x in triples if x[2] == False]
        return ["I ran into a problem:"]+arg_assumptions+arg_problems


class StateMachine():

    def __init__(self):
        self.iris = IRIS
        # be careful: instead of threading this through requests, keeping it here
        self.class_id = None

    def state_machine(self, data, prepend=[]):
        state = data["state"]
        if state == "START":
            return self.state_start(data)
        elif state == "CLASSIFICATION":
            return self.state_classify(data)
        elif state == "RESOLVE_ARGS":
            return self.state_resolve_args(data, prepend)
        elif state == "OPTIONS":
            return self.state_options(data)
        elif state == "SELECT_OPTION":
            return self.state_select_option(data)
        elif state == "EXECUTE":
            return self.state_execute(data, prepend)

    def state_start(self, data):
        messages = data["messages"]
        # since this is starting an interaction, we only need to look at the first/last message
        text = util.get_start_message(messages)
        class_id, class_text, pred = self.iris.get_predictions(text)[0]
        print(class_id, class_text, pred)
        # keep track of class_id so we can use it later
        self.class_id = class_id
        return { "state": "CLASSIFICATION", "text": ["Would you like to \"{}\"?".format(class_text[0])] }

    def state_classify(self, data):
        messages = data["messages"]
        text = util.get_classification_message(messages) # message that holds the user yes/no response
        verify = util.verify_response(text)
        # since user accepted this task we will try to resolve args now
        if verify:
            return self.state_machine({"state": "RESOLVE_ARGS", "messages":messages })
        # user doesn't want this, so start over
        else:
            return {"state": "OPTIONS", "text": ["Would you like similar options?"] }

    def state_options(self, data):
        messages = data["messages"]
        text = util.get_classification_message(messages) # message that holds the user yes/no response (RENAME THIS)
        verify = util.verify_response(text)
        if verify:
            initial_query = util.get_start_message(messages)
            predictions = self.iris.get_predictions(initial_query, 3)
            options = ["{}: \"{}\"".format(i,x[1][0]) for i,x in enumerate(predictions)]
            self.option_set = {i:x[0] for i,x in enumerate(predictions)}
            return {"state": "SELECT_OPTION", "text": ["Okay, here are some similar options:"]+options+["Would you like any of these?"] }
        else:
            return {"state": "START", "text": ["Okay, what would you like to do?"] }

    def state_select_option(self, data):
        messages = data["messages"]
        text = util.get_classification_message(messages)# last message (RENAME THIS)
        success, value = util.extract_number(text)
        if success:
            self.class_id = self.option_set[value]
            cmd_name = self.iris.class2cmd[self.class_id][0]
            return self.state_machine({"state": "RESOLVE_ARGS", "messages":messages }, ["Cool, I can \"{}\"".format(cmd_name)])
        else:
            return {"state": "START", "text": ["Okay, what would you like to do?"] }

    def state_resolve_args(self, data, prepend_message=[]):
        messages = data["messages"]
        class_id = self.class_id
        # the first message is the one we will be using for argument extraction from base command
        text = util.get_resolve_args_message(messages)
        function_data = self.iris.class_functions[class_id] # this gets types, args, and function executable
        cmd_names = self.iris.class2cmd[class_id] # list of possible command strings
        message_args = util.parse_message_args(messages)
        # case 1: all args can be inferred from matches with one of the command strings
        for cmd in cmd_names:
            succ, map_ = util.arg_match(text, cmd)
            if succ:
                if all([arg in map_ for arg in function_data["args"]]):
                    arg_list = [self.iris.magic_type_convert(map_[arg], function_data["types"][arg]) for arg in function_data["args"]]
                    return self.state_machine({"state": "EXECUTE", "messages": messages, "text": text, "arg_list": arg_list,
                                               "arg_names": function_data["args"]}, prepend_message)
                # so we got a successful mapping, but still need to ask for args
                # don't forget the args we've already got...
                else:
                    message_args = {**message_args, **map_}
        # case 2: all args are in arg_map derived from messages
        if all([arg in message_args for arg in function_data["args"]]):
            arg_list = [self.iris.magic_type_convert(message_args[arg], function_data["types"][arg]) for arg in function_data["args"]]
            return self.state_machine({"state": "EXECUTE", "messages": messages, "text": text, "arg_list": arg_list,
                                       "arg_names": function_data["args"]}, prepend_message)
        # case 3: ask user for some number of args
        for arg in function_data["args"]:
            if (not arg in message_args):
                question_messages = function_data["types"][arg].question(arg)
                return {"state": "RESOLVE_ARGS", "text": prepend_message+question_messages, "arg":arg}

    def state_execute(self, data, prepend_message=[]):
        class_id = self.class_id
        label = self.iris.class2title[class_id].upper()
        response_text = prepend_message + process_succ_failure(self.iris, class_id, data["arg_list"], data["arg_names"], data["text"])
        # if this was a recursed call, want to pop something off the stack here
        return {"state":"START", "text": response_text, "label":label }
