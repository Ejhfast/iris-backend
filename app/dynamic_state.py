import sys
import os
sys.path.insert(0, os.path.abspath('..'))
from iris import util

class BaseQuestion:
    def set_user_response(self, response):
        self.user_response = response

    def reset(self):
        self.user_response = None
        self.current_state = self

class YesNo(BaseQuestion):
    user_response = None
    def __init__(self, question, yes=None, no=None):
        self.question = question
        self.yes = yes
        self.no = no

    def next_state(self):
        if util.verify_response(self.user_response):
            return isinstance(self.yes, BaseQuestion), self.yes
        else:
            return isinstance(self.no, BaseQuestion), self.no

class Select(BaseQuestion):
    user_response = None
    def __init__(self, question, options={}):
        self.id2option = {}
        option_keys = sorted(options.keys())
        question_text = []
        question_text.append(question)
        for i,k in enumerate(option_keys):
            self.id2option[i] = options[k]
            question_text.append("{}: {}".format(i,k))
        self.question = "\n".join(question_text)
        self.current_state = self

    def next_state(self):
        success, choice = util.extract_number(self.user_response)
        if success:
            if choice in self.id2option:
                self.current_state = self.id2option[choice]
                return isinstance(self.current_state, BaseQuestion)
        # retry
        self.user_response = None
        return True

question_tree = YesNo("Is the data normally distributed? (If you're not sure, say 'no'.)",
    yes=YesNo("Do your samples have equal variance? (If you're not sure, say 'no'.)",
        yes="student t-test",
        no="welch t-test"
    ),
    no="mann-whitney u test"
)

select_tree = Select("Which of the following metrics are you interested in", options={
    "Accuracy: description of what accuracy is": "use the keyword accuracy",
    "F1: the geometric mean of precision and recall": Select("What kind of F1?", options={
        "Macro: average of F1 across all classes": "use the keyword f1_macro",
        "Micro: weighted average of F1 across classes": "use the keyword f1_micro",
        "Binary: F1 for only the positive class": "use the keyword f1"
    })
})

def recurse(machine):
    if (machine.user_response):
        keep_going, machine = machine.next_state()
        if not keep_going:
            print("Finished:", machine)
        else:
            recurse(machine)
    else:
        response = input(machine.question+"\n")
        machine.set_user_response(response)
        recurse(machine)

recurse(select_tree)
# machine = question_tree
#
# def recurse_two(state):
#     global machine
#     if state["state"] == "SPECIAL":
#         machine.set_user_response(state["user_response"])
#         keep_going, machine = machine.next_state()
#         if not keep_going:
#             return({"state":"DONE", "text":machine})
#         else:
#             return({"state":"SPECIAL", "text": machine.question})
#     else:
#         return "Not doing that now"
#
# print(recurse_two({"state":"SPECIAL", "user_response":"yes"}))
# print(recurse_two({"state":"SPECIAL", "user_response":"no"}))
