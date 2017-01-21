from iris import iris_objects
from iris import state_types as t
from iris import state_machine as sm
from iris import util
import json

class Done(sm.StateMachine):
    def __init__(self):
        super().__init__()
        self.output = ["Great, I've loaded in the dataframe."]
        self.accepts_input = False
    def next_state_base(self, next):
        file_name = self.context["ASSIGNMENT_NAMES"]["loaded_file"]
        dataframe = iris_objects.IrisDataframe(file_name, self.context["headers"], self.context["types"], self.context["data"])
        return True, sm.ValueState(dataframe).when_done(self.get_when_done_state())

class SetIndex(sm.StateMachine):
    def __init__(self, index, category_var):
        self.category_var = category_var
        self.index = index
        super().__init__()
    def next_state_base(self, text):
        self.context["types"][self.index] = self.category_var(self.context).get_value()
        return True, sm.DoAll([
            sm.Print([
                "Great, I've set {} to {}.".format(self.context["headers"][self.index], self.category_var.get_value()),
            ]),
            CheckTypes(force_check=True)
        ]).when_done(self.get_when_done_state())

class ChangeIndex(sm.StateMachine):
    def __init__(self):
        super().__init__()
        self.output = ["If you'd like to change something, give me the column index:"]
    def next_state_base(self, text):
        success, index = util.extract_number(text)
        column_names = {i:k for i,k in enumerate(self.context["headers"])}
        column_types = {i:t for i,t in enumerate(self.context["types"])}
        if success and index in column_names:
            category_var = sm.Variable("category_var")
            print_text = sm.Print([
                "What type would you like to change {} to?".format(column_names[index]) +
                " It's currently been set to a {} value.".format(column_types[index])
            ])
            set_index = SetIndex(index, category_var).when_done(self.get_when_done_state())
            select_types = sm.Assign(category_var, t.Select(options={
                "Number (e.g., 2 or 3.4)": "Number",
                "String (e.g., 'a line of text')": "String",
                "Categorical (e.g., 'large' or 'small')": "Categorical"
            }))
            return True, sm.DoAll([print_text, select_types]).when_done(set_index)
        return True, Done().when_done(self.get_when_done_state())

class ExamineTypes(sm.StateMachine):
    def __init__(self):
        super().__init__()
        self.output = ["Would you like to change anything?"]
    def next_state_base(self, text):
        if util.verify_response(text):
            return True, ChangeIndex().when_done(self.get_when_done_state())
        return True, Done().when_done(self.get_when_done_state())

def detect_data_type(data):
    try:
        float(data)
        return "Number"
    except:
        if len(data.split()) > 1:
            return "Text"
        else:
            return "Categorical"

def rows_and_types(cols):
    return [detect_data_type(x) for x in cols]

class CheckTypes(sm.StateMachine):
    def __init__(self, force_check=False):
        super().__init__()
        self.force_check = force_check
        self.output = ["Would you like to examine the automatically inferred types for each column?"]
        if self.force_check:
            self.output = []
            self.accepts_input = False
    def next_state_base(self, text):
        file_str = self.context['data']
        types = rows_and_types(file_str[0].split(","))
        if not self.force_check:
            self.context["types"] = types
        if self.force_check or util.verify_response(text):
            type_obj = {
                i: {
                    "name": self.context["headers"][i],
                    "type": self.context["types"][i],
                    "example": self.context["data"][0].split(",")[i]
                } for i,_ in enumerate(self.context["headers"])
            }
            print_types = sm.Print([{"type":"data", "value":json.dumps(type_obj, indent=4, default=str)}])
            return True, sm.DoAll([print_types, ChangeIndex()]).when_done(self.get_when_done_state())
        return True, Done().when_done(self.get_when_done_state())

class AskForHeaders(sm.StateMachine):
    def __init__(self):
        super().__init__()
    def get_output(self):
        print(self.context)
        start_from = 1 if self.context["ASSIGNMENTS"]["throw_away"] else 0
        sample_data = self.context["ASSIGNMENTS"]["loaded_file"].split("\n")[start_from].split(",")
        return [
            "What are the headers? Please enter a list of comma-separated values. I've provided a line of sample data below.",
            {"type":"data", "value":json.dumps(sample_data, indent=4, default=str)}
        ]
    def next_state_base(self, text):
        possible_headers = [x.strip() for x in text.split(",")]
        if len(possible_headers) == len(self.context['headers']):
            self.context['headers'] = possible_headers
            start_from = 1 if self.context["ASSIGNMENTS"]["throw_away"] else 0
            self.context["data"] = self.context["ASSIGNMENTS"]["loaded_file"].split("\n")[start_from:]
            return True, sm.Print(["Great, thanks."]).when_done(self.get_when_done_state())
        else:
            problem = sm.Print([
                "I ran into a problem. You need to enter {} values.".format(len(self.context["headers"]))
            ]).when_done(self)
        return True, problem

class GenerateHeaders(sm.StateMachine):
    def __init__(self):
        super().__init__()
        self.output = ["Great, I generated headers:"]
        self.accepts_input = False
    def next_state_base(self, text):
        file_str = self.context["ASSIGNMENTS"]["loaded_file"]
        lines = file_str.split("\n")
        num_cols = len(lines[0].split(","))
        headers = ["column{}".format(i) for i in range(0,num_cols)]
        self.context['headers'] = headers
        start_from = 1 if self.context["ASSIGNMNETS"]["throw_away"] else 0
        self.context['data'] = file_str.split("\n")[start_from:]
        format_header = json.dumps(headers, indent=4)
        # generate headers
        return True, sm.Print([{"type":"data", "value":format_header}]).when_done(self.get_when_done_state())

def check_file_header(file_str):
    first_line = file_str.split("\n")[0]
    cols = first_line.split(",")
    types = rows_and_types(cols)
    print(set(types))
    if len(set(types)) > 1:
        return False, cols
    return True, cols

class VerifyHeader(sm.StateMachine):
    def __init__(self, headers):
        self.headers = headers
        super().__init__()
        self.output = [
            "Here are the headers I inferred from the first line. Do these look good?",
            {"type":"data", "value":headers}
        ]
    def next_state_base(self, text):
        if util.verify_response(text):
            self.context['data'] = self.context["ASSIGNMENTS"]["loaded_file"].split("\n")[1:]
            return True, sm.Print(["Great, thanks."]).when_done(self.get_when_done_state())
        return True, CheckHeader(force_ask=True).when_done(self.get_when_done_state())

class CheckHeader(sm.StateMachine):
    def __init__(self, force_ask=False):
        self.force_ask = force_ask
        super().__init__()
        self.accepts_input = False
    def next_state_base(self, text):
        # check for header
        file_str = self.context["ASSIGNMENTS"]["loaded_file"]
        success, headers = check_file_header(file_str)
        self.context['headers'] = headers
        format_header = json.dumps(headers, indent=4)
        if not success or self.force_ask:
            self.context["ASSIGNMENTS"]["throw_away"] = False
            throw_away = sm.Assign("throw_away",
                t.YesNo("In that case, would you like to throw away the first line of data?",
                    yes=True,
                    no=False))
            print_result = sm.Print([
                "This file does not appear to have a header." if not self.force_ask else "Okay, how would you like to generate the header?"
            ])
            selection = t.Select(options={
                "Generate the values automatically": GenerateHeaders(),
                "Enter the headers manually": AskForHeaders(),
                "Use first line as header:": VerifyHeader(", ".join(headers))
            })
            if self.force_ask:
                return True, sm.DoAll([throw_away, print_result, selection]).when_done(self.get_when_done_state())
            else:
                return True, sm.DoAll([print_result, selection]).when_done(self.get_when_done_state())
        return True, VerifyHeader(format_header).when_done(self.get_when_done_state())


class Init(sm.StateMachine):
    accepts_input = False
    def next_state_base(self, text):
        return True, sm.Assign("loaded_file", t.File("What is the name of the file")).when_done(self.get_when_done_state())

file_state = sm.DoAll([
    Init(),
    CheckHeader(),
    CheckTypes()
])
