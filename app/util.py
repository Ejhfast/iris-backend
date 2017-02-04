from collections import defaultdict
import numpy as np
from sklearn.linear_model import LogisticRegression
from iris import iris_objects

# crappy data type inference function
def detect_data_type(data):
    try:
        return "Number", float(data)
    except:
        if len(data.split()) > 1:
            return "Text", data
        else:
            return "Categorical", data

# crappy function to figure out csv type and header
# TODO: deal with no header
# TODO: deal with quoted strings
def rows_and_types(data):
    title = data[0]
    cols = title.replace("\"","").split(",")
    sample_line = data[1].split(",")
    return list(zip(cols,[detect_data_type(x) for x in sample_line]))

def process_data(meta, content):
    env = defaultdict(list)
    cat2index = {}
    for j,line in enumerate(content):
        if j == 0: continue
        cols = line.replace("\"","").split(",")
        for i,col in enumerate(cols):
            val = None
            if meta[i]["type"] == "number":
                val = float(col)
            elif meta[i]["type"] == "categorical":
                if col in cat2index:
                    val = cat2index[col]
                else:
                    val = len(cat2index)
                    cat2index[col] = val
            else:
                val = col
            env[meta[i]["name"]].append(val)
    for k,vs in meta.items():
        tp = meta[k]["type"]
        if tp in ["number", "categorical"]:
            env[meta[k]["name"]] = np.array(env[meta[k]["name"]])
    return env

def detect_type(x):
    if isinstance(x, str):
        return "string"
    elif isinstance(x, int):
        return "int"
    elif isinstance(x, dict):
        return "dict"
    elif isinstance(x, float):
        return "float"
    elif isinstance(x, np.ndarray):
        return "array"
    elif isinstance(x, list):
        return "list"
    elif isinstance(x, iris_objects.IrisImage):
        return "image"
    elif isinstance(x, iris_objects.IrisModel):
        return "model"
    elif isinstance(x, iris_objects.IrisData):
        return "dataset"
    elif isinstance(x, iris_objects.IrisDataframe):
        return "dataframe"
    elif isinstance(x, iris_objects.FunctionWrapper):
        return "function"
    else:
        return str(type(x))

def env_vars(iris):
    out = []
    for k,v in iris.env.items():
        if k == "__MEMORY__": continue
        key = k.name if isinstance(k, iris_objects.IrisValue) else k
        out.append({"name": key, "value": detect_type(v), "order": iris.env_order[k]})
    return out

def make_xy(meta, env):
    X, y = None, None
    feature_index = {"X":{}, "y":{}}
    def init_or_append(X,data):
        npx = np.array(data)
        npx = npx.reshape(npx.shape[0],1)
        if X != None: return np.concatenate((X,npx),axis=1)
        return npx
    for k,vs in meta.items():
        if "x_value" in vs:
            X = init_or_append(X, env[vs["name"]])
            feature_index["X"][X.shape[1]-1] = vs["name"]
        if "y_value" in vs:
            y = init_or_append(y, env[vs["name"]])
            feature_index["y"][y.shape[1]-1] = vs["name"]
    if y.shape[1] == 1:
        y = y.reshape(y.shape[0])
    return X,y,feature_index
