from fastapi import FastAPI, UploadFile
from PIL import Image, ImageDraw, ImageFont
import io
from roboflow import Roboflow
from PIL import Image, ImageDraw, ImageFont
import uvicorn
import os
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

rf = Roboflow(api_key="1O1S51rNvyKVHyNejQyh")
project = rf.workspace().project("surfer-spotting")
model = project.version(2).model
app = FastAPI()

# Serve static files from the "static" directory
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/images_infered", StaticFiles(directory="images_infered"), name="images_infered")

def save_with_bbox_renders(img):
    file_name = os.path.basename(img.filename)
    path = './images_infered/'+ file_name
    img.save(path)
    return file_name

def draw_boxes(box, x0, y0, img, class_name, surfer_count):
    # OPTIONAL - color map, change the key-values for each color to make the
    # class output labels specific to your dataset
    color_map = {
        "class1":"red",
        "class2":"blue",
        "class3":"yellow",
        "class4":"green"
    }

    # get position coordinates
    bbox = ImageDraw.Draw(img)

    bbox.rectangle(box, outline="yellow", width=2)
    font = ImageFont.truetype("Fonts/BebasNeue-Regular.ttf", 10)
    bbox.text((x0+10, y0-5), class_name, fill='yellow', font=font, anchor='mm')
    
    text = f"Surfer Count {surfer_count}"
    font2 = ImageFont.truetype("Fonts/BebasNeue-Regular.ttf", 40)
    bbox.text((20, 40), text, fill='yellow', font=font2, anchor='lm')

    return img

@app.get("/")
async def read_root():
    return FileResponse("static/index.html")

@app.post("/process-image/")
async def process_image(file: UploadFile):
    # Create a temporary file to save the uploaded image
    with open("temp_image.png", "wb") as temp_image:
        temp_image.write(file.file.read())

    # Perform inference and annotation
    predictions, file_name = predict("temp_image.png")

    # Save the annotated image with the same filename in the "images_infered" directory
    
    # Gather additional information
    surfer_count = len(predictions)

    # Return the path to the annotated image (relative to the project directory) and information
    return {
        "surfer_count": surfer_count,
        "predictions": predictions,
        "annotated_image": file_name,
    }
def predict(image_path):
    predictions = model.predict(image_path, confidence=30, overlap=50).json()['predictions']

    newly_rendered_image = Image.open(image_path)
    file_name = None
    # RENDER
    # for each detection, create a crop and convert into CLIP encoding
    for prediction in predictions:
    # rip bounding box coordinates from current detection
    # note: infer returns center points of box as (x,y) and width, height
    # ----- but pillow crop requires the top left and bottom right points to crop
        x0 = prediction['x'] - prediction['width'] / 2
        x1 = prediction['x'] + prediction['width'] / 2
        y0 = prediction['y'] - prediction['height'] / 2
        y1 = prediction['y'] + prediction['height'] / 2
        box = (x0, y0, x1, y1)
        
        surfers_count = len(predictions)
        newly_rendered_image = draw_boxes(box, x0 , y0, newly_rendered_image, prediction['class'], surfers_count)
        file_name = save_with_bbox_renders(newly_rendered_image)
        
    return predictions, file_name

if __name__ == "__main__":
    # pip install python-multipart, fastapi, uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
