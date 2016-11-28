from aiohttp import web
import jinja2
import aiohttp_jinja2
import json
import os
import sys
import aiohttp_cors
from demo import iris
from iris import StateMachine
import util
from collections import defaultdict
from sklearn.linear_model import LogisticRegression

PORT = int(os.environ.get("PORT", 8000))

app = web.Application()
aiohttp_jinja2.setup(app, loader=jinja2.FileSystemLoader('templates/'))

cors = aiohttp_cors.setup(app)

content = None

state_machine = StateMachine(iris)

def add_cors(route):
    cors.add(route, {"*": aiohttp_cors.ResourceOptions(
                 allow_credentials=True,
                 expose_headers="*",
                 allow_headers="*")})

def parse_args(messages):
    print(messages)
    fail_indexes = [i for i,x in enumerate(messages) if x["origin"] == "iris" and x["type"] == "ask" ]
    args = {}
    for i in fail_indexes:
        iris_ask = messages[i]["text"]
        var = iris_ask.split()[-1][:-1]
        print(var)
        args[var] = messages[i+1]["text"]
    return args

# NEW FUNCS

async def process_csv(request):
    global content
    data = await request.post()
    csv = data['csv']
    filename = csv.filename
    content = csv.file.read().decode('utf-8').split('\n')
    data_types = util.rows_and_types(content)
    return web.Response(status=302, headers={"Location":"http://localhost:3000/select_data?data={}".format(json.dumps(data_types))})

add_cors(app.router.add_route('POST', '/upload_react', process_csv))

# END

async def new_loop(request):
    question = await request.json()
    print(question)
    response = state_machine.state_machine(question)
    response["origin"] = "iris"
    response["type"] = "ADD_SERVER_MESSAGE"
    print(response)
    return web.json_response(response)
    # question = question["messages"]
    # top_level_q = question[0]['text']
    # args = parse_args(question)
    # res = iris.loop(top_level_q, args)
    # if res[0] == "Success":
    #     results = {"response_type":"success", "type":"ADD_SERVER_MESSAGE", "origin":"iris", "text":res[1]}
    # elif res[0] == "Ask":
    #     results = {"response_type":"ask", "type":"ADD_SERVER_MESSAGE", "origin":"iris", "text":res[1]}
    # else:
    #     results = {"response_type":"fail", "type":"ADD_SERVER_MESSAGE", "origin":"iris", "text":res[1]}
    # print(results)
    # return web.json_response(results)

add_cors(app.router.add_route('POST', '/new_loop', new_loop))

async def import_data(request):
    data = await request.post()
    column_data = defaultdict(dict)
    for k,v in data.items():
        key = "_".join(k.split("_")[:-1])
        index = int(k.split("_")[-1])
        column_data[index][key] = v
    env = util.process_data(column_data, content)
    X,y,f = util.make_xy(column_data, env)
    iris.env["data_model"] = LogisticRegression()
    iris.env["features"] = X
    iris.env["classes"] = y
    iris.env["feature-table"] = f
    return web.Response(status=302, headers={"Location":"http://localhost:3000/"})

add_cors(app.router.add_route('POST', '/import_data', import_data))

async def classify_query(request):
    data = await request.post()
    query = data["query"]
    results = iris.best_n(query)
    return web.json_response(results)

add_cors(app.router.add_route('POST', '/classify', classify_query))

async def execute_function(request):
    data = await request.post()
    ex_class = data["class"]
    args = json.loads(data["args"])
    print(ex_class,args)
    execution = iris.call_class(int(ex_class), args)
    return web.json_response({"result": str(execution)})

add_cors(app.router.add_route('POST', '/execute', execute_function))

web.run_app(app,port=PORT)
