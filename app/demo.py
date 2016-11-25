import sys
import os
sys.path.insert(0, os.path.abspath('..'))
from iris import Iris, IrisValue, Int, IrisType, Any, List, String
from collections import defaultdict
from sklearn.linear_model import LogisticRegression
import fileinput
import numpy as np
iris = Iris()

# here we simply add two integers, result will be appended to
# result list in enviornment context
@iris.register("add {n1} and {n2}")
def add(n1 : Int, n2 : Int):
    return n1+n2

@iris.register("cross-validate using {score} and {n} folds")
def cross_validate(score : String, n : Int):
    model = iris.env["data_model"]
    x = iris.env["features"]
    y = iris.env["classes"]
    from sklearn.cross_validation import cross_val_score
    return cross_val_score(model, x, y, scoring = score, cv=n)


@iris.register("find the best regularization parameter")
def find_regularize():
    model = iris.env["data_model"]
    x = iris.env["features"]
    y = iris.env["classes"]
    from sklearn.cross_validation import cross_val_score
    import numpy as np
    best_score = 0
    best_c = None
    for c in [0.01, 0.1, 1, 10, 100]:
        model = LogisticRegression(C=c)
        score = np.average(cross_val_score(model, x, y, scoring = "accuracy", cv=5))
        if score > best_score:
            best_score = score
            best_c = c
    return "best l2 is {} with cross-validation performance of {} accuracy".format(best_c, best_score)

@iris.register("list possible features")
def list_features():
    x = iris.env["features"]
    features = list(iris.env["feature-table"]['X'].values())
    return features

@iris.register("find predictive value of {feature}")
def get_predictive_value(feature : String):
    model = iris.env["data_model"]
    x = iris.env["features"]
    y = iris.env["classes"]
    feature_table = iris.env["feature-table"]['X']
    f2i = {i:f for f,i in feature_table.items()}
    model.fit(x,y)
    return model.coef_[0][f2i[feature]]

@iris.register("predictive power of all features")
def all_features():
    model = iris.env["data_model"]
    x = iris.env["features"]
    y = iris.env["classes"]
    features = list(iris.env["feature-table"]['X'].values())
    feature_table = iris.env["feature-table"]['X']
    f2i = {i:f for f,i in feature_table.items()}
    model.fit(x,y)
    return "\n".join(["{} of {}".format(f,model.coef_[0][f2i[f]]) for f in features])

# so here we add a new named variable to enviornment context that
# holds the result
@iris.register("add {n1:Int} and {n2:Int} to var")
def add_named(n1 : Int, n2 : Int):
    return IrisValue(n1+n2, name="n1_and_n2")

# demonstrate lookup of variable from environment
@iris.register("sum {lst}")
def sum1(lst : List):
    return sum(lst)

@iris.register("count {lst}")
def count1(lst : Any):
    counts = defaultdict(int)
    for x in lst:
        counts[x] += 1
    return counts

@iris.register("make indicator for {lst}")
def make_indicator(lst : Any):
    keys = set(lst)
    index2key = {i:k for i,k in enumerate(keys)}
    key2index = {k:i for i,k in index2key.items()}
    return [key2index[x] for x in lst]

@iris.register("what vars")
def what_vars():
    return iris.env.keys()

@iris.register("last values")
def last_values():
    return iris.env["results"]

@iris.register("program enviornment")
def env():
    return iris.env

@iris.register("{x}")
def info(x : Any):
    return x

@iris.register("list commands")
def list_cmds():
    for k in iris.mappings.keys():
        print(k)

iris.env["my_list"] = [1,2,3,4,5]
iris.env["my_num"] = 3
model = LogisticRegression()
x_data = np.random.randint(0,10,size=(100,10))
y_data = (np.sum(x_data,axis=1) > 40).astype(int)
iris.env["data_model"] = model
iris.env["features"] = x_data
iris.env["classes"] = y_data


# data_cols = defaultdict(list)
# lookup = {}
# for i,line in enumerate(fileinput.input()):
#     if i == 0:
#         for j,col in enumerate(line.strip().split(",")):
#             lookup[j] = col
#     else:
#         for j,d in enumerate(line.strip().split(",")):
#             data_cols[lookup[j]].append(d)
#
# for k,vs in data_cols.items():
#     iris.env[k] = vs

iris.train_model()

init_message = [{"text": "add 3 4", "origin": "user"}]
state1 = iris.state_machine({ "state": "START", "messages":init_message})
print(state1)
state2 = iris.state_machine({"state": "CLASSIFICATION", "id": state1["id"], "messages":init_message+[{"text": "yes", "origin": "user"}]})
print(state2)

# iris.env_loop()
