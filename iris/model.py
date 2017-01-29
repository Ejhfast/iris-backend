from collections import defaultdict
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer

class IrisBase:

    def __init__(self):
        self.cmd2class = {}
        self.class2cmd = defaultdict(list)
        self.class_functions = {}
        self.model = LogisticRegression()
        self.vectorizer = CountVectorizer()
        self.env = {}
        self.env_order = {}
        self.history = {"history": [], "currentConvo": { 'messages': [], 'title': None, 'hidden': False, 'id': 0, 'args': {} }}

    def add_to_env(self, name, result):
        self.env[name] = result
        self.env_order[name] = len(self.env_order)

    def set_history(self, request):
        self.history = request["conversation"]

    def serialize_state(self):
        return {"env":self.env, "env_order":self.env_order, "history":self.history}

    def load_state(self, data):
        self.env = data["env"]
        self.env_order = data["env_order"]
        self.history = data["history"]

    def iris(self):
        return self

    def train_model(self):
        x_docs, y = zip(*[(k, v) for k,v in self.cmd2class.items()])
        x = self.vectorizer.fit_transform(x_docs)
        self.model.fit(x,y)

    def predict_input(self, query):
        return self.model.predict_proba(self.vectorizer.transform([query]))

    def learn_from_example(self, cls_idx, query_string, arg_triple):
        succs = [x[2] for x in arg_triple]
        print("learning")
        print(succs)
        if not all(succs):
            return False, None
        else:
            arg_map = {}
            for name,val,_ in arg_triple: arg_map[str(val)] = name
            transform = []
            query_words = query_string.lower().split()
            print(arg_map)
            for w in query_words:
                print(w, w in arg_map)
                if w in arg_map:
                    transform.append("{"+arg_map[w]+"}")
                else:
                    transform.append(w)
            command_string = " ".join(transform)
            if command_string in self.cmd2class: return False, None
            self.cmd2class[command_string] = cls_idx
            self.class2cmd[cls_idx].append(command_string)
            self.train_model()
            return True, command_string

    def get_predictions(self, text, n=1):
        predictions = self.predict_input(text)[0].tolist()
        sorted_predictions = sorted([(i,self.class2cmd[i],x) for i,x in enumerate(predictions)], key=lambda x: x[-1], reverse=True)
        return sorted_predictions[:n]

IRIS = IrisBase()
