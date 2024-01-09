import streamlit as st
from PIL import Image
import replicate
import os
import requests
import io
import openai
import numpy as np
from stl import mesh
import base64

# Ownership
st.text("\t\t\tAnton Varshavsky")
st.markdown("<hr>", unsafe_allow_html=True)  # Add a horizontal line

openai.api_key = # Your OpenAI API key here
os.environ["REPLICATE_API_TOKEN"] = # Your replicate token here

def get_character_for_brightness(brightness, brightness_scale, characters):
    normalized_brightness = brightness / 255
    closest_index = min(range(len(brightness_scale)), key=lambda i: abs(brightness_scale[i] - normalized_brightness))
    return characters[closest_index]

def to_braille(sentence):
    charToArray = {
        " ": [[0, 0], [0, 0], [0, 0]],
        ".": [[0, 0], [1, 1], [0, 1]],
        ",": [[0, 0], [1, 0], [0, 0]],
        "#": [[0, 1], [0, 1], [1, 1]],
        "a": [
            [1, 0],
            [0, 0],
            [0, 0]
        ],
        "b": [
            [1, 0],
            [1, 0],
            [0, 0]
        ],
        "c": [
            [1, 1],
            [0, 0],
            [0, 0]
        ],
        "d": [
            [1, 1],
            [0, 1],
            [0, 0]
        ],
        "e": [
            [1, 0],
            [0, 1],
            [0, 0]
        ],
        "f": [
            [1, 1],
            [1, 0],
            [0, 0]
        ],
        "g": [
            [1, 1],
            [1, 1],
            [0, 0]
        ],
        "h": [
            [1, 0],
            [1, 1],
            [0, 0]
        ],
        "i": [
            [0, 1],
            [1, 0],
            [0, 0]
        ],
        "j": [
            [0, 1],
            [1, 1],
            [0, 0]
        ],
        "k": [
            [1, 0],
            [0, 0],
            [1, 0]
        ],
        "l": [
            [1, 0],
            [1, 0],
            [1, 0]
        ],
        "m": [
            [1, 1],
            [0, 0],
            [1, 0]
        ],
        "n": [
            [1, 1],
            [0, 1],
            [1, 0]
        ],
        "o": [
            [1, 0],
            [0, 1],
            [1, 0]
        ],
        "p": [
            [1, 1],
            [1, 0],
            [1, 0]
        ],
        "q": [
            [1, 1],
            [1, 1],
            [1, 0]
        ],
        "r": [
            [1, 0],
            [1, 1],
            [1, 0]
        ],
        "s": [
            [0, 1],
            [1, 0],
            [1, 0]
        ],
        "t": [
            [0, 1],
            [1, 1],
            [1, 0]
        ],
        "u": [
            [1, 0],
            [0, 0],
            [1, 1]
        ],
        "v": [
            [1, 0],
            [1, 0],
            [1, 1]
        ],
        "w": [
            [0, 1],
            [1, 1],
            [0, 1]
        ],
        "x": [
            [1, 1],
            [0, 0],
            [1, 1]
        ],
        "y": [
            [1, 1],
            [0, 1],
            [1, 1]
        ],
        "z": [
            [1, 0],
            [0, 1],
            [1, 1]
        ],
    }

    punctuation = '!"$%&\'()*+-/:;<=>?@[\\]^_`{|}~'
    clean_string = ''.join(char for char in sentence if char not in punctuation)

    big_str = "\n\n\n\n\n"
    words = clean_string.split()

    line = ""
    for word in words:
        if len(line) + len(word) + 1 > 20:
            for _ in range(3):  # Repeat the line three times
                to_p = ''
                for letter in line:
                    for i in range(2):
                        to_p += str(charToArray[letter.lower()][_][i])
                big_str += f"{to_p}\n"
            big_str += "\n"  # Blank line after the three repeats
            line = ""  # Reset the line
        line += word + " "
    for _ in range(3):  # Repeat the last line three times
        to_p = ''
        for letter in line:
            for i in range(2):
                to_p += str(charToArray[letter.lower()][_][i])
        big_str += f"{to_p}\n"
    big_str += "\n"  # Blank line after the three repeats

    big_str = big_str.replace("1", ".")
    big_str = big_str.replace("0", " ")
    return big_str

def explain(prompt):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user",
                   "content": f"explain what {prompt} is in a simple way, keep in mind that this is being used for images so it is most likely not an advanced topic to explain so if there is a simpler synonym for the word then use that synonym for the explanation. For example, if the prompt is owl, it is reffering to the bird rather than the Web Language. Only use one-two sentances"}],
        temperature=.88
    )
    total_tokens_used_c = response['usage']['total_tokens']
    price = (total_tokens_used_c / 1000) * .0015
    st.session_state.actual_response = response['choices'][0]['message']['content']
    return st.session_state.actual_response

def create_img(prompt):
    output = replicate.run(
        "stability-ai/sdxl:c221b2b8ef527988fb59bf24a8b97c4561f1c671f73bd389f866bfb27c061316",
        input={"prompt": f"Create a minimalistic, flat design of a {prompt} in a uniform shade of white, set against a completely black background. The {prompt} must be fully white. Fully white. The {prompt} should have no texture or gradient, ensuring a single-tone appearance, suitable for ASCII (Braille) art conversion. The {prompt} should have a simple, distinct silhouette without any protruding or detailed elements, with a clear contrast against the background, yet seamlessly blending into the grayscale tone without any shading or highlighting. Just one main object in the center with no other detials. Complete black background"}
        # A single, clear representation of {prompt} centered on a pitch-black background. There should be no other objects, details, or distractions in the background or around the {prompt}. The entire surrounding should be pure black. THE BACKGROUND NEEDS NO DETAIL OTHER THAN PITCH BLACK"}#,grayscale on pitch black background, not too detailed, NO BACKGROUND DETAILS, nothing in the image but the main prompt"}
    )
    response = requests.get(output[0])
    image_content = io.BytesIO(response.content)

    #st.image(output)
    st.session_state.generated_img = output
    return image_content

def create_base_plane(width, height, depth=1.0, extension=1.5):
    # Increase width and shift vertices to the right
    width += extension
    vertices = np.array([
        [-extension, 0, 0], [width - extension, 0, 0], [width - extension, height, 0], [-extension, height, 0],  # Bottom vertices
        [-extension, 0, depth], [width - extension, 0, depth], [width - extension, height, depth], [-extension, height, depth]  # Top vertices
    ])

    # The faces remain the same
    faces = np.array([
        [0, 3, 1], [1, 3, 2], [0, 4, 7], [0, 7, 3], [4, 5, 6], [4, 6, 7],
        [5, 1, 2], [5, 2, 6], [2, 3, 6], [3, 7, 6], [0, 1, 5], [0, 5, 4]
    ])

    return vertices, faces


def ascii_to_stl_with_dots(input_file, output_file, base_depth=1.2, dot_diameter=1.44, dot_height=0.48):
    all_vertices = []
    all_faces = []

    with open(input_file, 'r') as file:
        lines = file.readlines()

    max_width = max(len(line.rstrip()) for line in lines) * 1.5
    total_height = len(lines) * 1.5

    # Create and add the base plane
    base_vertices, base_faces = create_base_plane(max_width, total_height, base_depth)
    all_vertices.extend(base_vertices)
    all_faces.extend(base_faces)

    radius_x = dot_diameter / 2
    radius_y = dot_diameter / 2
    radius_z = dot_height / 2

    # Offset for ASCII art dots (placing them on top of the base)
    offset_z = base_depth
    for y, line in enumerate(lines):
        for x, char in enumerate(line.rstrip()):
            if char != ' ':  # Ignore spaces
                center_x = x * 1.5
                center_y = (total_height - 1.5) - (y * 1.5)
                center_z = base_depth + radius_z

                ellipsoid_vertices, ellipsoid_faces = create_ellipsoid(
                    [center_x, center_y, center_z], radius_x, radius_y, radius_z
                )
                face_offset = len(all_vertices)
                all_vertices.extend(ellipsoid_vertices)
                all_faces.extend(ellipsoid_faces + face_offset)

    # Create the mesh
    combined_mesh = mesh.Mesh(np.zeros(len(all_faces), dtype=mesh.Mesh.dtype))
    for i, f in enumerate(all_faces):
        for j in range(3):
            combined_mesh.vectors[i][j] = all_vertices[f[j]]

    # Save to STL
    combined_mesh.save(output_file)


# Modified create_3d_rectangle function to include Z offset
def create_ellipsoid(center, radius_x, radius_y, radius_z, resolution=10):
    vertices = []
    faces = []

    # Generate vertices
    for i in range(resolution):
        theta = i * np.pi / (resolution - 1)
        for j in range(resolution):
            phi = j * 2 * np.pi / (resolution - 1)

            x = center[0] + radius_x * np.sin(theta) * np.cos(phi)
            y = center[1] + radius_y * np.sin(theta) * np.sin(phi)
            z = center[2] + radius_z * np.cos(theta)

            vertices.append([x, y, z])

    # Generate faces
    for i in range(resolution - 1):
        for j in range(resolution - 1):
            v1 = i * resolution + j
            v2 = v1 + resolution
            v3 = v1 + 1
            v4 = v2 + 1

            faces.append([v1, v2, v3])
            faces.append([v3, v2, v4])

    return np.array(vertices), np.array(faces)

def inversion_choice(width, height, w_step, h_step, img_gray):
    characters4 = "           .............:::::::::::::::::::::::::::::::::::::::::::::"
    characters5 = "           .............:::::::::::::::::::::::::::::::::::::::::::::"[::-1]
    colors = {}
    color = ""
    for y in range(0, height, h_step):
        for x in range(0, width, w_step):
            brightness = img_gray.getpixel((x, y))
            if brightness not in colors:
                colors[brightness] = 1
            else:
                colors[brightness] += 1
            color += f"{brightness}, "
    largest_entries = sorted(colors.items(), key=lambda x: x[1])[-2:]
    #st.write(largest_entries)

    if (largest_entries[0][0] >= largest_entries[1][0]):
        return characters5

    else:
        return characters4


if "actual_response" not in st.session_state:
    st.session_state.actual_response = ''

if "generated_img" not in st.session_state:
    st.session_state.generated_img = ''

if 'ascii_img' not in st.session_state:
    st.session_state.ascii_img = ''

if 'braille_img' not in st.session_state:
    st.session_state.braille_img = ''

if 'prompt' not in st.session_state:
    st.session_state.prompt = ''

if 'braille' not in st.session_state:
    st.session_state.braille = ''

if "stl_file_data" not in st.session_state:
    st.session_state.stl_file_data = ''

if "translated_braille" not in st.session_state:
    st.session_state.translated_braille = ''

if "stl_translated" not in st.session_state:
    st.session_state.stl_translated = ''

if 'page1_clicked' not in st.session_state:
    st.session_state['page1_clicked'] = False

if 'page2_clicked' not in st.session_state:
    st.session_state['page2_clicked'] = False

if "user_input" not in st.session_state:
    st.session_state.user_input = ''

def braille_translator():
    st.session_state.user_input = st.text_area("Your Text", st.session_state.user_input, height=300)
    if st.button("Submit"):
        st.session_state.translated_braille = to_braille(st.session_state.user_input)
        # Display the Braille output
        f = open("translated_braille.txt", "w")
        f.write(st.session_state.translated_braille)
        f.close()

        ascii_to_stl_with_dots("translated_braille.txt", 'translated.stl')

        with open("translated.stl", 'rb') as file:
            st.session_state.stl_translated = file.read()

        if st.session_state.translated_braille:

            st.markdown("---")
            st.download_button(
                label="Translated Braille",
                data=st.session_state.translated_braille,
                file_name="translated_braille.txt",
                mime="text/plain"
            )


            st.download_button(
                label="3D Translated File",
                data=st.session_state.stl_translated,
                file_name="translated.stl",
                mime="model/stl"
            )



def braille_page():
    prompt = st.text_input("What do you want to be converted to ascii art/braille?", st.session_state.prompt)

    #prompt = st.text_input("What do you want to be converted to ascii art?")
    if prompt != "":
        if st.button("Submit"):
            explain(prompt)
            st.session_state.braille = to_braille(st.session_state.actual_response)

            st.session_state.braille_img = ''
            st.session_state.ascii_img = ''
            # Given character and brightness scale lists
            characters = "    %8&WM#*oahkbdpqwmZO0QLCJUYXzcvunxrjft/\|()1{}[]?-_+~<>i!lI;:,\"^`\'."
            characters3 = "    ....................:::::::::::::::::::::::::::::::::::::::::::::"
            characters4 = "           .............:::::::::::::::::::::::::::::::::::::::::::::"
            characters5 = "           .............:::::::::::::::::::::::::::::::::::::::::::::"[::-1]
            characters2 = "    ....................:::::::::::::::::::::::::::::::::::::::::::::"[::-1]

            brightness_scale = [0, 0.0751, 0.0829, 0., 0.1227, 0.1403, 0.185, 0.2183, 0.2417, 0.2571, 0.2852, 0.2902, 0.2919, 0.3099, 0.3192, 0.3232, 0.3294, 0.3384, 0.3609, 0.3619, 0.3667, 0.3737, 0.3747, 0.3838, 0.3921, 0.396, 0.3984, 0.3993, 0.4075, 0.4091, 0.4101, 0.42, 0.4328, 0.4382, 0.4385, 0.442, 0.4473, 0.4477, 0.4503, 0.4562, 0.458, 0.461, 0.4638, 0.4667, 0.4686, 0.4693, 0.5509, 0.5567, 0.5569, 0.5591, 0.5602, 0.5602, 0.565, 0.5776, 0.6465,0.6595, 0.6631, 0.6714, 0.6759, 0.6809, 0.6925, 0.7039, 0.7086, 0.7235, 0.7302, 0.7602, 0.7834, 0.8037, 0.9999]

            # Open an image
            to_open = create_img(prompt)
            img = Image.open(to_open)

            # Convert the image to grayscale
            img_gray = img.convert('L')

            # ascii_img = ' '
            # braille_img = ' '
            print(len(characters), len(brightness_scale))

            # Get pixel data
            width, height = img_gray.size
            w_step = width//100
            h_step = height//50
            correct_character = inversion_choice(width, height, w_step, h_step, img_gray)
            for y in range(0, height, h_step):
                st.session_state.ascii_img += "\n"
                st.session_state.braille_img += "\n"
                for x in range(0, width, w_step):
                    brightness = img_gray.getpixel((x, y))
                    ascii_char = get_character_for_brightness(brightness, brightness_scale, characters)
                    st.session_state.ascii_img += ascii_char
                    brail_char = get_character_for_brightness(brightness, brightness_scale, correct_character) #invert here
                    st.session_state.braille_img += brail_char
                    print(f"Pixel ({x}, {y}): Brightness={brightness}, Character={ascii_char}")

            st.session_state.braille_img += f"\n{st.session_state.braille}"
            f = open("ascii.txt", "w")
            f.write(st.session_state.ascii_img)
            f.close()



            f = open("braille.txt", "w")
            f.write(st.session_state.braille_img)
            f.close()

            ascii_to_stl_with_dots("braille.txt", 'new.stl')



    if (st.session_state.ascii_img or st.session_state.braille_img):
        st.markdown("---")
        st.image(st.session_state.generated_img)
        st.write(st.session_state.actual_response)
        st.write("Braille can be seen in file")
        st.text(" ")

        st.markdown("---")
        st.download_button(
            label="Ascii Art File",
            data=st.session_state.ascii_img,
            file_name="ascii.txt",
            mime="text/plain"
        )

        st.download_button(
            label="Braille File (With Text)",
            data=st.session_state.braille_img,
            file_name="braille.txt",
            mime="text/plain"
        )

        with open("new.stl", 'rb') as file:
            st.session_state.stl_file_data = file.read()

        st.download_button(
            label="3D Print File",
            data=st.session_state.stl_file_data,
            file_name="new.stl",
            mime="model/stl"
        )
        st.text("Fullscreen the text file to see result")

        st.markdown("---")
        st.markdown("[3D Print Braille](https://text2stl.mestres.fr/en-us/generator?modelSettings=%7B%22fontName%22%3A%22Roboto%22%2C%22variantName%22%3A%22regular%22%2C%22text%22%3A%22Bienvenue%20!%22%2C%22size%22%3A45%2C%22height%22%3A10%2C%22spacing%22%3A2%2C%22vSpacing%22%3A0%2C%22alignment%22%3A%22center%22%2C%22vAlignment%22%3A%22default%22%2C%22type%22%3A2%2C%22supportHeight%22%3A5%2C%22supportBorderRadius%22%3A5%2C%22supportPadding%22%3A%22%7B%5C%22top%5C%22%3A10%2C%5C%22bottom%5C%22%3A10%2C%5C%22left%5C%22%3A10%2C%5C%22right%5C%22%3A10%7D%22%2C%22handleSettings%22%3A%22%7B%5C%22type%5C%22%3A%5C%22none%5C%22%2C%5C%22position%5C%22%3A%5C%22top%5C%22%2C%5C%22size%5C%22%3A10%2C%5C%22size2%5C%22%3A2%2C%5C%22offsetX%5C%22%3A0%2C%5C%22offsetY%5C%22%3A0%7D%22%7D)")
        st.write("""For better results, use these settings:
         Text alignment: Left
         Text vertical alignemnt: Default
         Text size: (make the field blank)
         Text height: 5mm
         Text kerning: -2mm
         Line kerning: -30mm
         """)


st.sidebar.title("Navigation")

if st.sidebar.button("Text to Image and Braille"):
    st.session_state['page1_clicked'] = True
    st.session_state['page2_clicked'] = False

if st.sidebar.button("Braille Translator"):
    st.session_state['page2_clicked'] = True
    st.session_state['page1_clicked'] = False


if st.session_state['page1_clicked']:
    braille_page()
if st.session_state['page2_clicked']:
    braille_translator()
