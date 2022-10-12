from fastapi import FastAPI
import gradio as gr
import uvicorn
import sys
# Usage: python3 embedd-gradio-fastapi-test.py 0.0.0.0 1234
CUSTOM_PATH = "/gradio"

app = FastAPI()

# homepage
@app.get("/")
def read_main():
    return {"message": "This is your main app"}

# gradio interface
io = gr.Interface(lambda x: "Hello, " + x + "!", "textbox", "textbox")
# create gradio app
gradio_app = gr.routes.App.create_app(io)
# mount gradio app into ./gradio
app.mount(CUSTOM_PATH, gradio_app)

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('Run `python embedd-gradio-fastapi-test.py <HOST> <PORT>`')
        sys.exit(1)

    host = sys.argv[1]
    port = int(sys.argv[2])

    uvicorn.run('embedd-gradio-fastapi-test:app', host=host, port=port)

