from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pathlib import Path

app = FastAPI()

# Mount the "templates" folder as a static directory
app.mount("/templates", StaticFiles(directory="templates"), name="templates")

@app.get("/")
def read_root():
    html_file = Path("templates/index.html")
    return FileResponse(html_file)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)

