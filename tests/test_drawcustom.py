import os
import pytest
from unittest.mock import patch, MagicMock
from PIL import Image, ImageChops
import io
from custom_components.open_epaper_link.imagegen import customimage, getIndexColor
current_dir = os.path.dirname(os.path.abspath(__file__))
print(current_dir)
BASE_IMG_PATH = os.path.join(current_dir, "test_images")
@pytest.fixture
def mock_hass():
    hass = MagicMock()
    hass.states.get.return_value = MagicMock(attributes={'width': 296, 'height': 128})
    hass.config.path.return_value = "img.png"
    return hass

@pytest.fixture
def mock_service():
    return MagicMock(data={
        'rotate': 0,
        'dither': False,
        'background': 'white',
        'payload': []
    })
def test_should_pass():
    assert True

def images_equal(img1, img2):
    """Compare two images and return True if they are identical."""
    return ImageChops.difference(img1, img2).getbbox() is None
def save_image(image):
    # save the generated image for debugging
    img_path = os.path.join(BASE_IMG_PATH, 'rename_me.png')
    with open(img_path, 'wb') as f:
        f.write(image)

# Text

def test_text_basic(mock_hass, mock_service):
    """
    Test basic text rendering with default settings.
    """
    mock_service.data['payload'] = [{
        'type': 'text',
        'x': 10,
        'y': 10,
        'value': 'Hello, World!',
        'size': 20,
        'color': 'black'
    }]

    result = customimage('entity_id', mock_service, mock_hass)
    generated_img = Image.open(io.BytesIO(result))
    example_img = Image.open(os.path.join(BASE_IMG_PATH, 'text_basic.png'))
    assert images_equal(generated_img, example_img), "Basic text rendering failed"

# Multiline text

# def test_text_multiline(mock_hass, mock_service):
#     """
#     Test multiline text rendering with custom font and color.
#     """
#     mock_service.data['payload'] = [{
#         'type': 'text',
#         'x': 10,
#         'y': 10,
#         'value': 'Hello,\nWorld!',
#         'size': 18,
#         'color': 'red',
#     }]
#
#     result = customimage('entity_id', mock_service, mock_hass)
#     generated_img = Image.open(io.BytesIO(result))
#     example_img = Image.open(os.path.join(BASE_IMG_PATH, 'text_multiline.png'))
#
#     assert images_equal(generated_img, example_img), "Multiline text rendering failed"

# Line

def test_line_basic(mock_hass, mock_service):
    """
    Test basic line drawing with default width.
    """
    mock_service.data['payload'] = [{
        'type': 'line',
        'x_start': 10,
        'y_start': 10,
        'x_end': 100,
        'y_end': 100,
        'fill': 'black'
    }]

    result = customimage('entity_id', mock_service, mock_hass)
    generated_img = Image.open(io.BytesIO(result))
    example_img = Image.open(os.path.join(BASE_IMG_PATH, 'line_basic.png'))

    assert images_equal(generated_img, example_img), "Basic line drawing failed"

def test_line_custom(mock_hass, mock_service):
    """
    Test line drawing with custom width and color.
    """
    mock_service.data['payload'] = [{
        'type': 'line',
        'x_start': 50,
        'y_start': 20,
        'x_end': 200,
        'y_end': 100,
        'fill': 'red',
        'width': 3
    }]

    result = customimage('entity_id', mock_service, mock_hass)
    generated_img = Image.open(io.BytesIO(result))
    example_img = Image.open(os.path.join(BASE_IMG_PATH, 'line_custom.png'))

    assert images_equal(generated_img, example_img), "Custom line drawing failed"

# Rectangle

def test_rectangle_filled(mock_hass, mock_service):
    """
    Test filled rectangle drawing.
    """
    mock_service.data['payload'] = [{
        'type': 'rectangle',
        'x_start': 50,
        'y_start': 20,
        'x_end': 200,
        'y_end': 100,
        'fill': 'blue',
        'outline': 'black',
        'width': 2
    }]

    result = customimage('entity_id', mock_service, mock_hass)
    generated_img = Image.open(io.BytesIO(result))
    example_img = Image.open(os.path.join(BASE_IMG_PATH, 'rectangle_filled.png'))

    assert images_equal(generated_img, example_img), "Filled rectangle drawing failed"

def test_rectangle_outline(mock_hass, mock_service):
    """
    Test rectangle drawing with only outline.
    """
    mock_service.data['payload'] = [{
        'type': 'rectangle',
        'x_start': 50,
        'y_start': 20,
        'x_end': 200,
        'y_end': 100,
        'outline': 'red',
        'width': 3
    }]

    result = customimage('entity_id', mock_service, mock_hass)
    generated_img = Image.open(io.BytesIO(result))
    example_img = Image.open(os.path.join(BASE_IMG_PATH, 'rectangle_outline.png'))

    assert images_equal(generated_img, example_img), "Rectangle outline drawing failed"

# Circle

def test_circle_filled(mock_hass, mock_service):
    """
    Test filled circle drawing.
    """
    mock_service.data['payload'] = [{
        'type': 'circle',
        'x': 100,
        'y': 64,
        'radius': 50,
        'fill': 'red',
        'outline': 'black',
        'width': 2
    }]

    result = customimage('entity_id', mock_service, mock_hass)
    generated_img = Image.open(io.BytesIO(result))
    example_img = Image.open(os.path.join(BASE_IMG_PATH, 'circle_filled.png'))

    assert images_equal(generated_img, example_img), "Filled circle drawing failed"

def test_circle_outline(mock_hass, mock_service):
    """
    Test circle drawing with only outline.
    """
    mock_service.data['payload'] = [{
        'type': 'circle',
        'x': 100,
        'y': 64,
        'radius': 50,
        'outline': 'red',
        'width': 3
    }]

    result = customimage('entity_id', mock_service, mock_hass)
    generated_img = Image.open(io.BytesIO(result))
    example_img = Image.open(os.path.join(BASE_IMG_PATH, 'circle_outline.png'))

    assert images_equal(generated_img, example_img), "Circle outline drawing failed"

# Multiple elements

def test_multiple_elements(mock_hass, mock_service):
    """
    Test drawing multiple elements on the same image.
    """
    mock_service.data['payload'] = [
        {'type': 'rectangle', 'x_start': 0, 'y_start': 0, 'x_end': 296, 'y_end': 128, 'fill': 'white'},
        {'type': 'text', 'x': 10, 'y': 10, 'value': 'Hello', 'size': 20, 'color': 'black'},
        {'type': 'line', 'x_start': 0, 'y_start': 40, 'x_end': 296, 'y_end': 40, 'fill': 'black', 'width': 1},
        {'type': 'circle', 'x': 148, 'y': 84, 'radius': 30, 'fill': 'red'}
    ]

    result = customimage('entity_id', mock_service, mock_hass)
    generated_img = Image.open(io.BytesIO(result))
    example_img = Image.open(os.path.join(BASE_IMG_PATH, 'multiple_elements.png'))

    assert images_equal(generated_img, example_img), "Multiple elements drawing failed"

# Rotation

def test_rotation(mock_hass, mock_service):
    """
    Test image rotation.
    """
    mock_service.data['rotate'] = 90
    mock_service.data['payload'] = [{
        'type': 'text',
        'x': 10,
        'y': 10,
        'value': 'Rotated',
        'size': 20,
        'color': 'black'
    }]

    result = customimage('entity_id', mock_service, mock_hass)
    generated_img = Image.open(io.BytesIO(result))
    example_img = Image.open(os.path.join(BASE_IMG_PATH, 'rotated.png'))

    assert images_equal(generated_img, example_img), "Image rotation failed"