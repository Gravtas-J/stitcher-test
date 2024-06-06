import streamlit as st
from PIL import Image, ImageOps, ImageEnhance, ImageFont, ImageDraw
import numpy as np
from sklearn.cluster import KMeans
from reportlab.lib.pagesizes import landscape, letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import tempfile
import os
from math import ceil
import zipfile
import pandas as pd
# from pyth.plugins.rtf15.reader import Rtf15Reader
# from pyth.plugins.plaintext.writer import PlaintextWriter
import platform
# Set the limit to a higher value to avoid DecompressionBombError
Image.MAX_IMAGE_PIXELS = None

def pixelate_image(image, size):
    small_image = image.resize((size, size), Image.NEAREST)
    result = small_image.resize(image.size, Image.NEAREST)
    return result

def resize_image_aspect_ratio(image, max_size):
    original_width, original_height = image.size
    aspect_ratio = original_width / original_height

    if original_width > original_height:
        new_width = max_size
        new_height = int(new_width / aspect_ratio)
    else:
        new_height = max_size
        new_width = int(new_height * aspect_ratio)

    resized_image = image.resize((new_width, new_height), Image.NEAREST)
    return resized_image

def apply_filter(image, filter_type):
    if filter_type == 'Greyscale':
        return ImageOps.grayscale(image)
    elif filter_type == 'Sepia':
        sepia_image = ImageOps.colorize(ImageOps.grayscale(image), '#704214', '#C0C080')
        return sepia_image
    elif filter_type == 'RGB Scale':
        return ImageEnhance.Color(image).enhance(2)
    else:
        return image

def reduce_colors(image, num_colors):
    if image.mode != 'RGB':
        image = image.convert('RGB')
    arr = np.array(image)
    h, w, c = arr.shape
    arr = arr.reshape((-1, 3))

    kmeans = KMeans(n_clusters=num_colors, random_state=42)
    labels = kmeans.fit_predict(arr)
    palette = kmeans.cluster_centers_.astype(int)

    result = palette[labels].reshape((h, w, 3)).astype(np.uint8)
    return Image.fromarray(result)

def darken_color(color, factor=0.7):
    return tuple(int(c * factor) for c in color)

def color_to_hex(color):
    return "#{:02x}{:02x}{:02x}".format(color[0], color[1], color[2])

def create_pattern_pdf(image_path, key_file_path):
    pdf_filename = os.path.join(tempfile.gettempdir(), "cross_stitch_pattern.pdf")
    c = canvas.Canvas(pdf_filename, pagesize=landscape(letter))
    
    # Detect the operating system and set the font path accordingly
    system = platform.system()
    if system == "Windows":
        font_path = "C:/Windows/Fonts/DejaVuSans-Bold.ttf"
    elif system == "Darwin":  # macOS
        font_path = "/Library/Fonts/DejaVuSans-Bold.ttf"
    else:  # Assume Linux or other UNIX-like OS
        font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    
    # Check if the font path exists
    if not os.path.exists(font_path):
        raise FileNotFoundError(f"Font not found at {font_path}")
    
    pdfmetrics.registerFont(TTFont('DejaVuSans-Bold', font_path))
    c.setFont('DejaVuSans-Bold', 10)
    
    # Load the image
    image = Image.open(image_path)
    image_reader = ImageReader(image)
    
    # Get image size
    img_width, img_height = image.size
    
    # Define page size
    page_width, page_height = landscape(letter)
    
    # Calculate the number of pages needed
    x_pages = ceil(img_width / page_width)
    y_pages = ceil(img_height / page_height)
    
    for y in range(y_pages):
        for x in range(x_pages):
            # Calculate the cropping box
            left = x * page_width
            top = y * page_height
            right = min((x + 1) * page_width, img_width)
            bottom = min((y + 1) * page_height, img_height)
            
            cropped_image = image.crop((left, top, right, bottom))
            cropped_image_reader = ImageReader(cropped_image)
            
            # Calculate position to place the image on the page
            x_pos = (page_width - (right - left)) / 2
            y_pos = (page_height - (bottom - top)) / 2
            
            # Draw the image on the canvas
            c.drawImage(cropped_image_reader, x_pos, y_pos, width=(right - left), height=(bottom - top))
            
            c.showPage()
    
    # Add the key text file to the PDF
    with open(key_file_path, 'r', encoding='utf-8') as key_file:
        lines = key_file.readlines()
        num_lines_per_page = 40
        num_pages = ceil(len(lines) / num_lines_per_page)
        for page in range(num_pages):
            c.showPage()
            c.setFont('DejaVuSans-Bold', 10)
            for i, line in enumerate(lines[page*num_lines_per_page:(page+1)*num_lines_per_page]):
                c.drawString(72, page_height - 72 - i*12, line.strip())
    
    c.save()
    return pdf_filename

def create_key_pdf(key_file_path):
    pdf_key_filename = os.path.join(tempfile.gettempdir(), "color_key.pdf")
    c = canvas.Canvas(pdf_key_filename, pagesize=letter)
    
    # Detect the operating system and set the font path accordingly
    system = platform.system()
    if system == "Windows":
        font_path = "C:/Windows/Fonts/DejaVuSans-Bold.ttf"
    elif system == "Darwin":  # macOS
        font_path = "/Library/Fonts/DejaVuSans-Bold.ttf"
    else:  # Assume Linux or other UNIX-like OS
        font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    
    # Check if the font path exists
    if not os.path.exists(font_path):
        raise FileNotFoundError(f"Font not found at {font_path}")
    
    pdfmetrics.registerFont(TTFont('DejaVuSans-Bold', font_path))
    c.setFont('DejaVuSans-Bold', 10)
    
    with open(key_file_path, 'r', encoding='utf-8') as key_file:
        lines = key_file.readlines()
        y = 750  # Start position for text
        for line in lines:
            c.drawString(72, y, line.strip())
            y -= 12  # Move down for the next line
            if y < 50:  # Check if we need a new page
                c.showPage()
                c.setFont('DejaVuSans-Bold', 10)
                y = 750
    
    c.save()
    
    return pdf_key_filename

def create_cross_stitch_key(symbol_map, closest_color_map, dmc_color_map, file_path):
    seen_symbols = set()
    with open(file_path, 'w', encoding='utf-8') as file:
        for color, symbol in symbol_map.items():
            if symbol not in seen_symbols:
                closest_color = closest_color_map[color]
                dmc_info = dmc_color_map[closest_color]
                hex_color = color_to_hex(closest_color)
                file.write(f"{symbol}: DMC {dmc_info['DMC']}, Name: {dmc_info['Floss Name']}, Hex: {hex_color}, RGB: {closest_color}\n")
                seen_symbols.add(symbol)

def load_dmc_colors(csv_file_path):
    dmc_colors = pd.read_csv(csv_file_path)
    color_map = {}
    for _, row in dmc_colors.iterrows():
        color = (row['Red'], row['Green'], row['Blue'])
        color_map[color] = {
            'DMC': row['DMC'],
            'Floss Name': row['Floss Name'],
            'Hex Code': row['Hex Code'],
        }
    return color_map

def find_closest_color(color, color_map):
    color_array = np.array(list(color_map.keys()))
    distances = np.linalg.norm(color_array - color, axis=1)
    closest_color_index = np.argmin(distances)
    closest_color = tuple(color_array[closest_color_index])
    return closest_color

def create_cross_stitch_image(image, dmc_color_map, symbol_size=1, max_colors=489):
    symbols = "▪▫▬▲►▼◄◊○●◘◙◦☺☻☼♀♂!$%&()*+<=>?@{|}~¡¢£¤¥¦§©«®±µ¶˄˅ѲѺ♠♥♣♦♪♯ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqurstuvwxyz12345678♔♕♗♘♙⚲⚳→➔░▒▓▔▖▘▙▚▛▜▝▞▟■□▢▣▤▥▦▧▨▩▪▫☜▭▮☞▰▱☝△▴▵▶▷▸▹►▻▼▽▾▿◀◁◂◃◄◅◆◇◈◉◊○◌◍◎●◐◑◒◓◔◕◖◗◘◙◚◛◜◝◞◟◠◡◢◣◤◥◦◧◨◩◪◫◬◭◮☟◰◱◲◳◴◵◶◷◸◹◺☖☘☙☚◿☀☁☂☃☄★☆☇☈☉☊☋☌☍☎☏☐☑☒⌦⌬①③⑥②④⑤⑦⑧⑨⑩♳♴♵♶♷♸♹♺⚀⚁⚂⚃⚄⚅⚆⚇⚈⚉⚐⚑⚒⚓⚔⚕⚖⚗⚘⚙⚚⚛⚜⚠⚡⚭⚮⛀⛁⛂⛃✂✄✆✇✈✉✌✍✎✏✐✑✒✓✧✩✪✫✬✯✿❀❄❥⧐⧑⧒⨂⨍⨖⪋⪌⪍⪎⪏⪐⪑⪒⪓⪔⪕⪖⪗⪘⪙⪚⪛⪜⪝⪞⪟⪠⪮⪯⪰⪱⪲⪳⪴⪵⪶⪷⪸⪹⪺⫹⫺⬀⬁⬂⬃⬄⬅⬆ꙔꙖꙢꙤꙦꙨꙪꙮꚋꜺףּצּרּתּﺚ⠻⠸⠶⠴⠱⡙⡛⡝⡟⡡⡣⡥⠽⠿⠗⠕⢍⢋⢉⢇⢅⢃⢁⡿⡽⢥⢧⢩⢫⢭⢯⢱⢳⣉⣈⣇⣆⣅⣄⣃⣁⣀⢿⢽⢼⢺⣎⣐⣒⣔⣖⣘⣚⣜⣞⣠⣢⣣⣤⣥⣩⣫⪁⪕⪏⪎⪌⪋⪊⪈ⱷⵥʥʫʠΩάήΰβδζθλνφψϊόϐϓϔϖ"
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"     
    font = ImageFont.truetype(font_path, symbol_size)
    reduced_image = reduce_colors(image, num_colors=max_colors)
    
    arr = np.array(reduced_image)
    symbol_map = {}
    closest_color_map = {}
    color_used = {}
    for color in np.unique(arr.reshape(-1, arr.shape[2]), axis=0):
        closest_color = find_closest_color(color, dmc_color_map)
        if closest_color not in color_used:
            color_used[closest_color] = symbols[len(color_used) % len(symbols)]
        symbol_map[tuple(color)] = color_used[closest_color]
        closest_color_map[tuple(color)] = closest_color
    
    cross_stitch_img = Image.new('RGB', (arr.shape[1] * symbol_size, arr.shape[0] * symbol_size), 'white')
    draw = ImageDraw.Draw(cross_stitch_img)
    
    for y in range(arr.shape[0]):
        for x in range(arr.shape[1]):
            color = tuple(arr[y, x])
            symbol = symbol_map[color]
            darkened_color = darken_color(color)
            text_x = x * symbol_size + symbol_size // 2
            text_y = y * symbol_size + symbol_size // 2
            draw.text((text_x, text_y), symbol, font=font, fill=darkened_color, anchor='mm')
            draw.rectangle([x * symbol_size, y * symbol_size, (x + 1) * symbol_size, (y + 1) * symbol_size], outline='black')
    
    for y in range(0, arr.shape[0], 10):
        draw.line([(0, y * symbol_size), (arr.shape[1] * symbol_size, y * symbol_size)], fill='red', width=1)
    for x in range(0, arr.shape[1], 10):
        draw.line([(x * symbol_size, 0), (x * symbol_size, arr.shape[0] * symbol_size)], fill='red', width=1)
    
    center_x = (arr.shape[1] // 2) * symbol_size
    center_y = (arr.shape[0] // 2) * symbol_size
    draw.line([(center_x, 0), (center_x, arr.shape[0] * symbol_size)], fill='green', width=3)
    draw.line([(0, center_y), (arr.shape[1] * symbol_size, center_y)], fill='green', width=3)
    
    cross_stitch_img_path = os.path.join(tempfile.gettempdir(), "cross_stitch_image.png")
    cross_stitch_img.save(cross_stitch_img_path)
    
    key_file_path = os.path.join(tempfile.gettempdir(), "color_key.txt")
    create_cross_stitch_key(symbol_map, closest_color_map, dmc_color_map, key_file_path)
    
    return cross_stitch_img_path, key_file_path

def bundle_files_into_zip(image_path, pdf_path, key_file_path, key_pdf_path):
    zip_path = os.path.join(tempfile.gettempdir(), "cross_stitch_pattern.zip")
    with zipfile.ZipFile(zip_path, 'w') as zipf:
        zipf.write(image_path, os.path.basename(image_path))
        zipf.write(pdf_path, os.path.basename(pdf_path))
        zipf.write(key_file_path, os.path.basename(key_file_path))
        zipf.write(key_pdf_path, os.path.basename(key_pdf_path))
        # zipf.write(key_rtf_path, os.path.basename(key_rtf_path))
    return zip_path

# st.logo("logo.png")
st.sidebar.title("Stitcher")
uploaded_file = st.sidebar.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "webp"])
pixelation_option = st.sidebar.checkbox("Pixelate Image", value=True)

# Add a slider for pixelation size if pixelation option is selected
if pixelation_option:
    pixelation_size = st.sidebar.select_slider("Select Pixelation Size", options=[4, 8, 16, 32, 64, 128, 256, 512], value=128)
cross_stitch_option = st.sidebar.checkbox("Generate Cross Stitch Pattern bundle", value=True)
if cross_stitch_option:
    max_colors = st.sidebar.select_slider("Max Colors for Cross Stitch", options=[4, 8, 16, 32, 64, 128, 256, 489], value=128)

filter_option = st.sidebar.selectbox("Select Filter", ["None", "Greyscale", "Sepia"])
#color_option = st.sidebar.selectbox("Select Color Palette", ["Original", "4 Colors", "8 Colors", "16 Colors"])
process_button = st.sidebar.button("Process Image")

if uploaded_file and process_button:
    image = Image.open(uploaded_file)
    st.image(image, caption='Original Image', use_column_width=True)

    max_colors = int(max_colors)

    if pixelation_option:
        sizes = [pixelation_size]  # Use the selected pixelation size
        pixelated_images = []
        zip_paths = []

        for size in sizes:
            pixelated_image = pixelate_image(image, size)
            filtered_image = apply_filter(pixelated_image, filter_option)
            
            pixelated_images.append(filtered_image)

            if cross_stitch_option:
                resized_image = resize_image_aspect_ratio(image, size)
                dmc_color_map = load_dmc_colors("DMC_RGB.csv")  # Load DMC colors from the provided CSV file
                combined_image_path, key_file_path = create_cross_stitch_image(resized_image, dmc_color_map, symbol_size=20, max_colors=max_colors)
                pdf_filename = create_pattern_pdf(combined_image_path, key_file_path)
                key_pdf_filename = create_key_pdf(key_file_path)
                # key_rtf_filename = create_key_rtf(key_file_path)
                zip_path = bundle_files_into_zip(combined_image_path, pdf_filename, key_file_path, key_pdf_filename)
                zip_paths.append(zip_path)

        col1, col2, col3, col4 = st.columns(4)

        for i, col in enumerate([col1]):
            with col:
                st.image(pixelated_images[i], caption=f'{sizes[i]}x{sizes[i]} - {filter_option}', use_column_width=True)
                st.download_button(label=f"Download {sizes[i]}x{sizes[i]}", data=pixelated_images[i].tobytes(), file_name=f"pixelated_{sizes[i]}x{sizes[i]}.png", mime="image/png", key=f"download_button_{i}")
                
                if cross_stitch_option and zip_paths:
                    with open(zip_paths[i], "rb") as file:
                        st.download_button("Download Cross Stitch Bundle", file, file_name="cross_stitch_pattern.zip", mime="application/zip", key=f"bundle_download_button_{i}")
    else:
        if cross_stitch_option:
            dmc_color_map = load_dmc_colors("DMC_RGB.csv")  # Load DMC colors from the provided CSV file
            combined_image_path, key_file_path = create_cross_stitch_image(image, dmc_color_map, symbol_size=20, max_colors=max_colors)
            pdf_filename = create_pattern_pdf(combined_image_path, key_file_path)
            key_pdf_filename = create_key_pdf(key_file_path)
            # key_rtf_filename = create_key_rtf(key_file_path)
            zip_path = bundle_files_into_zip(combined_image_path, pdf_filename, key_file_path, key_pdf_filename)
            
            st.image(Image.open(combined_image_path), caption='Cross Stitch Pattern', use_column_width=True)
            with open(zip_path, "rb") as file:
                st.download_button("Download Cross Stitch Bundle", file, file_name="cross_stitch_pattern.zip", mime="application/zip")