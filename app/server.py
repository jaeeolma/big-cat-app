from starlette.applications import Starlette
from starlette.responses import HTMLResponse, JSONResponse
from starlette.exceptions import HTTPException
from starlette.staticfiles import StaticFiles
from starlette.middleware.cors import CORSMiddleware
import uvicorn, aiohttp, asyncio
from io import BytesIO

from fastai.vision import *

model_file_name = 'big_cats_fastai_stage1_resnet34'
classes = ['Caracal',
           'Cheetah', 
           'Clouded leopard', 
           'Cougar', 
           'Jaguar', 
           'Jungle cat',
           'Leopard',
           'Lion', 
           'Lynx',
           'Serval',
           'Snow leopard', 
           'Sunda clouded leopard', 
           'Tiger']

path = Path(__file__).parent
print(path)

app = Starlette()
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_headers=['X-Requested-With', 'Content-Type'])
app.mount('/static', StaticFiles(directory='app/static'))

async def download_file(url, dest):
    if dest.exists(): return
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            print('downloading')
            data = await response.read()
            with open(dest, 'wb') as f: f.write(data)

async def setup_learner():
    #print('downloading')
    #await download_file(model_file_url, path/'models'/f'{model_file_name}.pth')
    data_bunch = ImageDataBunch.single_from_classes(path, classes,
        ds_tfms=get_transforms(), size=224).normalize(imagenet_stats)
    learn = create_cnn(data_bunch, models.resnet34, pretrained=False)
    learn.load(model_file_name)
    return learn

loop = asyncio.get_event_loop()
tasks = [asyncio.ensure_future(setup_learner())]
learn = loop.run_until_complete(asyncio.gather(*tasks))[0]
loop.close()

@app.route('/')
def index(request):
    html = path/'view'/'index.html'
    return HTMLResponse(html.open().read())

@app.route('/analyze', methods=['POST'])
async def analyze(request):
    data = await request.form()
    img_bytes = await (data['file'].read())
    img = open_image(BytesIO(img_bytes))
    res = learn.predict(img)
    top_3 = (res[2].numpy()).argsort()[-3:][::-1]
    resp_dict = {'best_result': str(res[0]),
                'best_confidence': round(float(res[2][top_3[0]]), 3),
                'second_result': classes[top_3[1]],
                'second_confidence': round(float(res[2][top_3[1]]), 3),
                'third_result': classes[top_3[2]],
                'third_confidence': round(float(res[2][top_3[2]]), 3)}
    return JSONResponse(resp_dict)

@app.route('/analyze_url', methods=['GET'])
async def analyze_url(request):
    try:
        img_bytes = await get_bytes(request.query_params["url"])
        img = open_image(BytesIO(img_bytes))
        res = learn.predict(img)
        top_3 = (res[2].numpy()).argsort()[-3:][::-1]
        resp_dict = {'best_result': str(res[0]),
                    'best_confidence': round(float(res[2][top_3[0]]), 3),
                    'second_result': classes[top_3[1]],
                    'second_confidence': round(float(res[2][top_3[1]]), 3),
                    'third_result': classes[top_3[2]],
                    'third_confidence': round(float(res[2][top_3[2]]), 3)}
    except:
        raise HTTPException(500, detail='Invalid image URL')
    return JSONResponse(resp_dict)

async def get_bytes(url):
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.read()

def start_app():
    uvicorn.run(app, host='0.0.0.0', port=8080)

if __name__ == '__main__':
    if 'serve' in sys.argv: uvicorn.run(app, host='0.0.0.0', port=8080)

