import streamlit as st
from stqdm import stqdm
from PIL import Image
import os
from optimizer import *

os.environ['KMP_DUPLICATE_LIB_OK']='True'

st.set_page_config(layout="wide")
st.markdown("<h1 style='text-align: center;'>Brushstroke Parameterized Style Transfer</h1>", unsafe_allow_html=True)

upload_content_img = None
upload_style_img = None

# side bar
st.sidebar.markdown("<h1 style='font-size: 20pt'><b>Upload Image</b></h1>", unsafe_allow_html=True)
# content image
st.sidebar.markdown("<p style='font-size: 15pt'><b>Content Image</b></p>", unsafe_allow_html=True)
upload_content_img = st.sidebar.file_uploader("Upload Content Image", type=["png", "jpg"])

# style image
st.sidebar.markdown("<p style='font-size: 15pt'><b>Style Image</b></p>", unsafe_allow_html=True)
upload_style_img = st.sidebar.file_uploader("Upload Style Image", type=["png", "jpg"])

# parameters
st.sidebar.markdown("<h1 style='font-size: 20pt'><b>Adjust Parameters</b></h1>", unsafe_allow_html=True)
# brushstroke optimization steps
n_steps_stroke = st.sidebar.slider("Brushstroke Optimization Steps", 20, 100, 100)
# pixel optimization steps
n_steps_pixel = st.sidebar.slider("Pixel Optimization Steps", 100, 5000, 2000)
# number of brushstrokes
n_strokes = st.sidebar.slider("Number of Brushstrokes", 1000, 10000, 5000)
# content weight
content_weight = st.sidebar.slider("Content Weight", 1.0, 50.0, 1.0)
# style weight -> brushstroke optimizer 기준
style_weight = st.sidebar.slider("Style Weight", 1.0, 50.0, 3.0)
# stroke width
stroke_width = st.sidebar.slider("Stroke Width", 0.01, 2.0, 0.1)
# stroke length
stroke_length = st.sidebar.slider("Stroke Length", 0.1, 2.0, 1.1)


# show image
# initialize
content_img = None
style_img = None
default_img = Image.open("./img/default.png")
default_img = default_img.resize((400, 400))

# split into 2 columns
col1, col2 = st.columns(2)

col1.header("Content Image")
if upload_content_img is not None:
    content_img = Image.open(upload_content_img)
    width, height = content_img.size
    col1.image(content_img, use_column_width=True, channels="RGB")
else:
    col1.image(default_img)

col2.header("Style Image")
if upload_style_img is not None:
    style_img = Image.open(upload_style_img)
    col2.image(style_img, use_column_width=True, channels="RGB")
else:
    col2.image(default_img)

# save tmp file
if content_img is not None and style_img is not None:
    if not os.path.exists(".tmp"):
        os.mkdir(".tmp")
    content_img.save(".tmp/content_img.png")
    style_img.save(".tmp/style_img.png")

stylize = st.button("Stylize!")

if (content_img is None or style_img is None) and stylize:
        st.warning("Upload Image First")

if stylize:
    # brushstroke optimization
    st.text("Start: Brushstroke Optimization...")
    pbar = stqdm(range(n_steps_stroke))
    stroke = BrushstrokeOptimizer(content_img, style_img, n_strokes=n_strokes, n_steps=n_steps_stroke,
                        width_scale=stroke_width, length_scale=stroke_length, 
                        content_weight=content_weight, style_weight=style_weight, streamlit_pbar=pbar)

    # optimize
    canvas = stroke.optimize()
    # pixel optimization
    st.text("Start: Pixel Optimization...")
    pbar = stqdm(range(n_steps_pixel))
    pixel = PixelOptimizer(canvas, content_img, style_img, n_steps=n_steps_pixel, style_weight=style_weight, 
                            content_weight=content_weight, streamlit_pbar=pbar)
    # optimize
    canvas = pixel.optimize()
    canvas = torch.squeeze(canvas).permute(1, 2, 0)
    canvas = canvas.cpu().detach().numpy()
    result = Image.fromarray(np.array(np.clip(canvas, 0, 1) * 255, dtype=np.uint8))
    # result    
    st.text("Done: Stylized Image is Ready!")
    st.image(result)

    # save in tmp folder
    result.save(".tmp/result.png")


    with open(".tmp/result.png", 'rb') as f:
        download_btn = st.download_button(
            label="Download Image",
            data=f,
            file_name="result.png",
            mime="image/png"
        )