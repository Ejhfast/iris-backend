from . import util
from ..model import IRIS_MODEL
from ... import state_machine as sm
from . import iris_objects
from .basic import EnvVar
# for statistical machine
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer

IRIS = IRIS_MODEL

class Dataframe(EnvVar):
    def is_type(self, x):
        if isinstance(x, iris_objects.IrisDataframe):
            return True
        return False
    def error_message(self, text):
        return ["I could not find dataframe {} in the environment.".format(text)]
    def type_from_string(self, text):
        return False, None

class DataframeSelector(sm.AssignableMachine):
    def __init__(self, question, dataframe = None):
        super().__init__()
        self.question = question
        self.dataframe = dataframe
        self.accepts_input = False
        if self.dataframe:
            self.write_variable("dataframe", dataframe)
            self.accepts_input = True
    def reset(self):
        self.accepts_input = False
        if self.dataframe:
            self.write_variable("dataframe", dataframe)
            self.accepts_input = True
        return self
    def get_output(self):
        dataframe = self.read_variable("dataframe")
        if dataframe:
            return [
                "Here are the current columns in the dataframe",
                {"type": "data", "value": util.prettify_data(dataframe.column_names)},
                "Please give me a comma-separated list of the columns you want to use for {}:".format(self.arg_name)
            ]
        return []
    def convert_type(self, x):
        return False, None
    def selector_transform(self, columns):
        dataframe = self.read_variable("dataframe")
        return np.array([dataframe.get_column(name) for name in columns]).T
    def next_state_base(self, text):
        if not self.read_variable("dataframe"):
            self.accepts_input = True
            return sm.Assign("dataframe", Dataframe(self.question)).when_done(self)
        else:
            dataframe = self.read_variable("dataframe")
            possible_columns = [x.strip() for x in text.split(",")]
            if all([col in dataframe.column_names for col in possible_columns]):
                selection = self.selector_transform(possible_columns)
                self.assign(selection)
                dataframe = self.delete_variable("dataframe")
                return selection
            return sm.Print(["A least one of those wasn't a valid column name. Try again?"]).when_done(self)

class DataframeNameSelector(DataframeSelector):
    def selector_transform(self, columns):
        dataframe = self.read_variable("dataframe")
        return (dataframe, columns)

class YesNo(sm.AssignableMachine):
    def __init__(self, question, yes=None, no=None):
        self.yes = yes
        self.no = no
        super().__init__()
        if isinstance(question, list):
            self.output = question
        else:
            self.output = [question]

    def string_representation(self, value):
        if isinstance(value, str) or isinstance(value, int):
            return str(value)
        return "CHOICE FOR {}".format(self.arg_name)

    def convert_type(self, text):
        return OR([
            primitive_or_question(self.yes, text),
            primitive_or_question(self.no, text)
        ])

    def base_hint(self, text):
        if util.verify_response(text):
            return ["triggers yes"]
        return ["triggers no"]

    def next_state_base(self, text):
        new_state = self
        if util.verify_response(text): new_state = self.yes
        else: new_state = self.no
        if not isinstance(new_state, sm.StateMachine):
            self.assign(new_state)
        return new_state

    def when_done(self, state):
        if isinstance(self.yes, sm.StateMachine):
            self.yes.when_done(state)
        if isinstance(self.no, sm.StateMachine):
            self.no.when_done(state)
        self.when_done_state = state
        return self

class Select(sm.AssignableMachine):
    def __init__(self, options={}, option_info={}, default=None):
        super().__init__()
        self.default = default
        self.id2option = {}
        option_keys = sorted(options.keys())
        question_text = []
        #question_text.append()
        for i,k in enumerate(option_keys):
            self.id2option[i] = options[k]
            question_text.append("{}: {}".format(i,k))
            if options[k] in option_info:
                for m in option_info[options[k]]:
                    question_text.append({"type":"explain", "value":m})
        question_text.append("Would you like any of these?")
        self.output = question_text

    def string_representation(self, value):
        if isinstance(value, str):
            return value
        return "CHOICE FOR {}".format(self.arg_name)

    def get_output(self):
        if self.arg_name != None:
            message = "Please choose from one of the following for {}:".format(self.arg_name)
            return [message] + self.output
        return ["Please choose from one of the following:"] + self.output

    def error_message(self, text):
        return ["{} is not a valid option".format(text)]

    def convert_type(self, text, doing_match=False):
        return OR([primitive_or_question(value, text, doing_match) for _, value in self.id2option.items()])

    def base_hint(self, text):
        success, choice = util.extract_number(text)
        if success:
            value = self.id2option[choice]
            if isinstance(value, str):
                return ["{}".format(value)]
            return ["choice {}".format(choice)]
        return []

    def next_state_base(self, text):
        new_state = self
        success, choice = util.extract_number(text)
        if success:
            if choice in self.id2option:
                new_state = self.id2option[choice]
                if not isinstance(new_state, sm.StateMachine):
                    self.assign(new_state, new_state)
                return new_state
        return self.set_error(self.error_message(text))

    def when_done(self, next_state):
        for id, state in self.id2option.items():
            if isinstance(state, sm.StateMachine):
                state.when_done(next_state)
        self.when_done_state = next_state
        return self

class StatisticalState(sm.AssignableMachine):
    def __init__(self, question, class2example):
        self.class2example = class2example
        self.titles = {}
        super().__init__()
        if isinstance(question, list):
            self.output = question
        else:
            self.output = [question]
        self.model = LogisticRegression()
        self.vectorizer = CountVectorizer()
        self.train()

    def train(self):
        docs = []
        classes = []
        self.transitions = {}
        for i, title in enumerate(self.class2example.keys()):
            examples = self.class2example[title]["examples"]
            self.transitions[i] = self.class2example[title]["state"]
            self.titles[i] = title
            for example in examples:
                docs.append(example)
                classes.append(i)
        X = self.vectorizer.fit_transform(docs)
        self.model.fit(X, classes)

    def base_hint(self, text):
        next_state = self.predict(text)
        return [self.titles[next_state]]

    def predict(self, text):
        x = self.vectorizer.transform([text])
        return self.model.predict(x)[0]

    def next_state_base(self, text):
        next_state = self.predict(text)
        if not isinstance(self.transitions[next_state], sm.StateMachine):
            self.assign(self.transitions[next_state])
        return self.transitions[next_state]

class Memory(sm.AssignableMachine):
    def __init__(self, iris = IRIS):
        self.iris = IRIS
        super().__init__()
        self.accepts_input = False
    def next_state_base(self, text):
        self.assign(iris_objects.EnvReference("__MEMORY__"))
        return iris_objects.EnvReference("__MEMORY__")
