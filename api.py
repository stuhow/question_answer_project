import fastapi



app = fastapi()

# Define a root `/` endpoint
@app.get('/')
def index():
    return {'ok': True}


@app.get('/predict')
def predict(Some_input):
    return {'wait': Some_input}
