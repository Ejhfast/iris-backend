from . import util
from .core import IRIS
from . import iris_types as t
from . import dynamic_state as ds

def process_succ_failure2(iris, cmd_object, arg_map, query):
    results = []
    for_ex = [(arg, arg_map[arg], True) for arg in cmd_object.all_args]
    args = [arg_map[arg] for arg in cmd_object.all_args]
    learn, lcmd = iris.learn_from_example(cmd_object.get_class_index(), query, for_ex)
    if learn:
        results.append("(I learned how to \"{}\")".format(lcmd))
    result = cmd_object.wrap_command(*args)
    iris.train_model()
    explanations = cmd_object.wrap_explanation(result)
    if isinstance(explanations, list):
        for explanation in explanations:
            results.append(explanation)
    else:
        results.append(explanations)
    return results


def process_succ_failure(iris, cls_idx, s_args, arg_names, query):
    succs = [x[0] for x in s_args]
    args = [x[1] for x in s_args]
    values = [x[2] for x in s_args]
    triples = list(zip(arg_names,args,succs))
    for_ex = list(zip(arg_names,values,succs))
    if all(succs):
        results = []
        learn, lcmd = iris.learn_from_example(cls_idx, query, for_ex)
        if learn:
            print("I learned")
            results.append("(I learned how to \"{}\")".format(lcmd))
        result = iris.class_functions[cls_idx].wrap_command(*args)
        iris.train_model()
        # potentially add some storage messages here
        # if isinstance(result, t.IrisImage):
        #     results.append({"type": "image", "value": result.value})
        # elif util.is_data(result):
        #     results.append({"type": "data", "value": util.prettify_data(result)})
        # else:
        explanations = iris.class_functions[cls_idx].wrap_explanation(result)
        if isinstance(explanations, list):
            for explanation in explanations:
                results.append(explanation)
        else:
            results.append(explanations)
        print(results)
        return results
    else:
        # kind of obnoxious that I am doing string construction here...
        assump = lambda x,y: "I think {} is \"{}\"".format(x,y)
        problem = lambda x,y: "For {}, {}".format(x,y)
        arg_assumptions = [assump(x[0],x[1]) for x in triples if x[2] == True]
        arg_problems = [problem(x[0],x[1]) for x in triples if x[2] == False]
        return ["I ran into a problem:"] + arg_assumptions + arg_problems


class StateMachine2:
    def __init__(self, iris = IRIS):
        self.machine = ds.StateMachineRunner(ds.IrisMachine())
        self.iris = iris
    def state_machine(self, data):
        outputs = []
        if not "first_call" in data:
            print("in not first call")
            text = util.get_last_message(data["messages"])
            keep_going = self.machine.next_state(text)
            # if we're done
            if not keep_going:
                print("ending it")
                out = self.machine.current_state
                return {"state":"START", "text": outputs }
            else:
                print("in alt not first call")
                for o in self.machine.current_output():
                    outputs.append(o)
                keep_going, more_outputs = self.machine.run_until_input_required()
                if not keep_going:
                    print("ending alt")
                    # this all needs to happen via APIs
                    class_id = self.machine.current_state.context["ASSIGNMENTS"]["user_class"]
                    cmd_object = self.iris.class_functions[class_id]
                    label = cmd_object.title.upper()
                    arg_map = {arg: self.machine.current_state.context["ASSIGNMENT_NAMES"][arg] for arg in cmd_object.all_args}
                    self.machine.reset()
                    return {"state":"START", "text": outputs + more_outputs, "label":label, "arg_map": arg_map }
                return {"state": "RECURSE", "text": outputs + more_outputs}
        else:
            print("in first call")
            outputs.append(self.machine.current_output())
            keep_going, more_outputs = self.machine.run_until_input_required()
            if not keep_going:
                print("ending first call")
                return {"state":"START", "text": outputs }
            return {"state": "RECURSE", "text": outputs + more_outputs}


class StateMachine():

    def __init__(self):
        self.iris = IRIS
        # self.machine = ds.select_tree
        # be careful: instead of threading this through requests, keeping it here
        self.class_id = None
        # hmm
        self.collect_args = {}
        self.working_arg = None
        self.machine = None

    def state_machine(self, data, prepend=[]):
        print(data)
        state = data["state"]
        if state == "START":
            return self.state_start(data)
        elif state == "CLASSIFICATION":
            return self.state_classify(data)
        elif state == "RESOLVE_ARGS":
            return self.state_resolve_args(data, prepend)
        elif state == "OPTIONS":
            return self.state_options(data)
        elif state == "EXPLAIN":
            return self.state_explain(data)
        elif state == "SELECT_OPTION":
            return self.state_select_option(data)
        elif state == "EXECUTE":
            return self.state_execute(data, prepend)
        elif state == "RECURSE":
            return self.state_recurse(data)

    def state_start(self, data):
        # return self.state_machine({"state": "RECURSE", "first_call": True})
        messages = data["messages"]
        # since this is starting an interaction, we only need to look at the first/last message
        text = util.get_start_message(messages)
        class_id, class_text, pred = self.iris.get_predictions(text)[0]
        print(class_id, class_text, pred)
        # keep track of class_id so we can use it later
        self.class_id = class_id
        return { "state": "CLASSIFICATION", "text": ["Would you like to {}?".format(class_text[0])] }

    def state_classify(self, data):
        messages = data["messages"]
        text = util.get_last_message(messages) # message that holds the user yes/no response
        if util.verify_response(text):
            # since user accepted this task we will try to resolve args now
            return self.state_machine({"state": "RESOLVE_ARGS", "messages":messages })
        elif util.verify_explain(text):
            class_id = self.class_id
            cmd_object = self.iris.class_functions[class_id]
            if cmd_object.help_text:
                explanation = [{"type": "explain", "value": text_} for text_ in cmd_object.help_text]
                return { "state": "EXPLAIN", "text": ["Here is some information about the command:"] + explanation + ["Would you like to continue?"]}
            else:
                response_ = ["I'm sorry, I don't have more information about this function.", "Would you like to continue?"]
                return { "state": "EXPLAIN", "text": response_ }
        # user doesn't want this, so start over
        else:
            return {"state": "OPTIONS", "text": ["Would you like similar options?"] }

    def state_options(self, data):
        messages = data["messages"]
        text = util.get_last_message(messages)
        verify = util.verify_response(text)
        if verify:
            initial_query = util.get_start_message(messages)
            predictions = self.iris.get_predictions(initial_query, 3)
            options = ["{}: \"{}\"".format(i,x[1][0]) for i,x in enumerate(predictions)]
            self.option_set = {i:x[0] for i,x in enumerate(predictions)}
            return {"state": "SELECT_OPTION", "text": ["Okay, here are some similar options:"] + options + ["Would you like any of these?"] }
        else:
            return {"state": "START", "text": ["Okay, what would you like to do?"] }

    def state_explain(self, data):
        messages = data["messages"]
        text = util.get_last_message(messages)
        if util.verify_response(text):
            return self.state_machine({"state": "RESOLVE_ARGS", "messages": messages})
        else:
            return {"state": "OPTIONS", "text": ["Would you like similar options?"] }

    def state_select_option(self, data):
        messages = data["messages"]
        text = util.get_last_message(messages)
        success, value = util.extract_number(text)
        if success:
            self.class_id = self.option_set[value]
            cmd_name = self.iris.class_functions[self.class_id].title
            return self.state_machine({"state": "RESOLVE_ARGS", "messages":messages }, ["Cool, I can \"{}\"".format(cmd_name)])
        elif util.verify_explain(text):
            initial_query = util.get_start_message(messages)
            predictions = self.iris.get_predictions(initial_query, 3)
            out = []
            for i, prediction in enumerate(predictions):
                out.append("{}: \"{}\"".format(i, prediction[1][0]))
                cmd_object = self.iris.class_functions[prediction[0]]
                if cmd_object.help_text:
                    for explanation in cmd_object.help_text:
                        out.append({"type": "explain", "value": explanation})
                else:
                    out.append({"type": "explain", "value": "Sorry, I don't have any explanation for this command."})
            return {"state": "SELECT_OPTION", "text": out + ["Would you like any of these options?"]}
        else:
            return {"state": "START", "text": ["Okay, what would you like to do?"] }

    def state_resolve_args(self, data, prepend_message=[]):
        messages = data["messages"]
        class_id = self.class_id
        # the first message is the one we will be using for argument extraction from base command
        text = util.get_resolve_args_message(messages)
        cmd_object = self.iris.class_functions[class_id] # this gets types, args, and function executable
        cmd_names = self.iris.class2cmd[class_id] # list of possible command strings
        message_args = util.parse_message_args(messages)
        message_args = util.transform_select_map(message_args, cmd_object, self.iris.env)
        message_args = {**message_args, **self.collect_args}
        # case 1: all args can be inferred from matches with one of the command strings
        for cmd in cmd_names:
            succ, map_ = util.arg_match(text, cmd)
            if succ:
                if all([arg in map_ for arg in cmd_object.all_args]):
                    arg_list = [self.iris.magic_type_convert(map_[arg], cmd_object.argument_types[arg]) for arg in cmd_object.all_args]
                    self.machine = None
                    self.collect_args = {}
                    self.working_arg = None
                    return self.state_machine({"state": "EXECUTE", "messages": messages, "text": text, "arg_list": arg_list,
                                               "arg_names": cmd_object.all_args, "arg_map": map_}, prepend_message)
                # so we got a successful mapping, but still need to ask for args, don't forget the args we've already got...
                else:
                    message_args = {**message_args, **map_}
        # case 2: all args are in arg_map derived from messages
        if all([arg in message_args for arg in cmd_object.all_args]):
            arg_list = [self.iris.magic_type_convert(message_args[arg], cmd_object.argument_types[arg]) for arg in cmd_object.all_args]
            self.machine = None
            self.collect_args = {}
            self.working_arg = None
            return self.state_machine({"state": "EXECUTE", "messages": messages, "text": text, "arg_list": arg_list,
                                       "arg_names": cmd_object.all_args, "arg_map": message_args}, prepend_message)
        # case 3: ask user for some number of args
        for arg in cmd_object.all_args:
            if (not arg in message_args):
                self.working_arg = arg
                self.machine = ds.StateMachineRunner(cmd_object.argument_types[arg])
                # question_messages = cmd_object.argument_types[arg].current_question()
                return self.state_machine({"state": "RECURSE", "first_call": True})
                # return {"state": "RESOLVE_ARGS", "text": prepend_message+question_messages, "arg":arg, "arg_map": message_args}

    # next step: tweak this and collect all args here!
    def state_recurse(self, data):
        outputs = []
        if not "first_call" in data:
            print("in not first call")
            text = util.get_last_message(data["messages"])
            keep_going = self.machine.next_state(text)
            # if we're done
            if not keep_going:
                print("ending it")
                out = self.machine.current_state
                self.machine.reset()
                self.collect_args[self.working_arg] = out.value
                print(out.context)
                return self.state_machine({"state":"RESOLVE_ARGS", "messages": data["messages"]})
            else:
                print("in alt not first call")
                for o in self.machine.current_output():
                    outputs.append(o)
                keep_going, more_outputs = self.machine.run_until_input_required()
                if not keep_going:
                    print("ending alt")
                    out = self.machine.current_state
                    self.machine.reset()
                    self.collect_args[self.working_arg] = out.value
                    return self.state_machine({"state":"RESOLVE_ARGS", "messages": data["messages"]}, prepend = outputs + more_outputs )
                return {"state": "RECURSE", "text": outputs + more_outputs}
        else:
            print("in first call")
            outputs.append(self.machine.current_output())
            keep_going, more_outputs = self.machine.run_until_input_required()
            if not keep_going:
                print("ending first call")
                return self.state_machine({"state":"RESOLVE_ARGS", "messages": data["messages"]})
            return {"state": "RECURSE", "text": outputs + more_outputs}

    def state_execute(self, data, prepend_message=[]):
        class_id = self.class_id
        label = self.iris.class_functions[class_id].title.upper()
        response_text = prepend_message + process_succ_failure(self.iris, class_id, data["arg_list"], data["arg_names"], data["text"])
        return {"state":"START", "text": response_text, "label":label, "arg_map": data["arg_map"] }
