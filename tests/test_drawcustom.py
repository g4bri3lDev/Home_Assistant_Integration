import os
from datetime import datetime

import pytest
from unittest.mock import patch, MagicMock
from PIL import Image, ImageChops
import io

from custom_components.open_epaper_link.imagegen import customimage, getIndexColor
from homeassistant.exceptions import HomeAssistantError

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

# def test_text_alignment(mock_hass, mock_service):
#     """
#     Test text alignment options (left, center, right).
#     """
#     mock_service.data['payload'] = [
#         {'type': 'text', 'y': 10, 'value': 'Left', 'align': 'left'},
#         {'type': 'text', 'x': 148, 'y': 40, 'value': 'Center', 'align': 'center'},
#         {'type': 'text', 'x': 148, 'y': 70, 'value': 'Right', 'align': 'right'}
#     ]
#     with pytest.raises(Exception):
#         result = customimage('entity_id', mock_service, mock_hass)
    # TODO fix this test
    # generated_img = Image.open(io.BytesIO(result))
    # example_img = Image.open(os.path.join(BASE_IMG_PATH, 'text_alignment.png'))
    #
    # assert images_equal(generated_img, example_img), "Text alignment failed"

def test_small_font_size(mock_hass, mock_service):
    """Test rendering text with very small font size."""
    mock_service.data['payload'] = [
        {'type': 'text', 'x': 10, 'y': 10, 'value': 'Tiny Text', 'size': 3}
    ]
    result = customimage('entity_id', mock_service, mock_hass)

    generated_img = Image.open(io.BytesIO(result))
    example_img = Image.open(os.path.join(BASE_IMG_PATH, 'small_font.png'))
    assert images_equal(generated_img, example_img), "Small font rendering failed"

def test_large_font_size(mock_hass, mock_service):
    """Test rendering text with very large font size."""
    mock_service.data['payload'] = [
        {'type': 'text', 'x': 10, 'y': 10, 'value': 'Huge', 'size': 150}
    ]
    result = customimage('entity_id', mock_service, mock_hass)
    generated_img = Image.open(io.BytesIO(result))
    example_img = Image.open(os.path.join(BASE_IMG_PATH, 'large_font.png'))
    assert images_equal(generated_img, example_img), "Large font rendering failed"

# Multiline text

# def test_text_alignment_specific(mock_hass, mock_service):
#     """
#     Test text alignment options (left, center, right) with a visible boundary.
#     This test creates a rectangle for each text to clearly show the alignment.
#     """
#     width = 296  # Assuming this is the width of your display
#     mock_service.data['payload'] = [
#         # Left aligned text
#         {'type': 'rectangle', 'x_start': 0, 'y_start': 0, 'x_end': width, 'y_end': 30, 'outline': 'black', 'width': 1},
#         {'type': 'text', 'x': 0, 'y': 15, 'value': 'Left Aligned', 'size': 20, 'color': 'black', 'align': 'left'},
#
#         # Center aligned text
#         {'type': 'rectangle', 'x_start': 0, 'y_start': 40, 'x_end': width, 'y_end': 70, 'outline': 'black', 'width': 1},
#         {'type': 'text', 'x': width // 2, 'y': 55, 'value': 'Center Aligned', 'size': 20, 'color': 'black', 'align': 'center'},
#
#         # Right aligned text
#         {'type': 'rectangle', 'x_start': 0, 'y_start': 80, 'x_end': width, 'y_end': 110, 'outline': 'black', 'width': 1},
#         {'type': 'text', 'x': width, 'y': 95, 'value': 'Right Aligned', 'size': 20, 'color': 'black', 'align': 'right'}
#     ]
#
#     result = customimage('entity_id', mock_service, mock_hass)
#     save_image(result)
#     generated_img = Image.open(io.BytesIO(result))
#     example_img = Image.open(os.path.join(BASE_IMG_PATH, 'text_alignment_specific.png'))
#
#     assert images_equal(generated_img, example_img), "Text alignment failed"

def test_text_wrapping(mock_hass, mock_service):
    """
    Test automatic text wrapping within a specified width.
    """
    mock_service.data['payload'] = [{
        'type': 'text',
        'x': 10,
        'y': 10,
        'value': 'This is a long text that should wrap to multiple lines automatically',
        'size': 16,
        'color': 'black',
        'max_width': 200
    }]

    result = customimage('entity_id', mock_service, mock_hass)
    generated_img = Image.open(io.BytesIO(result))
    example_img = Image.open(os.path.join(BASE_IMG_PATH, 'text_wrapping.png'))

    assert images_equal(generated_img, example_img), "Text wrapping failed"

def test_text_with_special_characters(mock_hass, mock_service):
    """Test rendering text with special characters."""
    mock_service.data['payload'] = [
        {'type': 'text', 'x': 10, 'y': 10, 'value': 'Special chars:', 'size': 20},
        {'type': 'text', 'x': 10, 'y': 30, 'value': 'áéíóú ñ ¿¡ @#$%^&*', 'size': 20}
    ]
    result = customimage('entity_id', mock_service, mock_hass)
    generated_img = Image.open(io.BytesIO(result))
    example_img = Image.open(os.path.join(BASE_IMG_PATH, 'text_special_chars.png'))

    assert isinstance(result, bytes)
    assert images_equal(generated_img, example_img), "Special characters failed"

# def test_text_with_extremely_long_string(mock_hass, mock_service):
#     """Test handling of extremely long text strings."""
#     long_text = "This is a very long string " * 100
#     mock_service.data['payload'] = [
#         {'type': 'text', 'x': 10, 'y': 10, 'value': long_text, 'size': 20}
#     ]
#     result = customimage('entity_id', mock_service, mock_hass)
#     # TODO fix wrapping
#     assert isinstance(result, bytes)

def test_text_multiline(mock_hass, mock_service):
    """
    Test multiline text rendering with custom font and color.
    """
    mock_service.data['payload'] = [{
        'type': 'text',
        'x': 10,
        'y': 10,
        'value': 'Hello,\nWorld!',
        'size': 18,
        'color': 'red',
    }]

    result = customimage('entity_id', mock_service, mock_hass)
    generated_img = Image.open(io.BytesIO(result))
    example_img = Image.open(os.path.join(BASE_IMG_PATH, 'text_multiline.png'))

    assert images_equal(generated_img, example_img), "Multiline text rendering failed"

def test_multiline_text(mock_hass, mock_service):
    """
    Test multiline text with custom delimiter and offset.
    """
    mock_service.data['payload'] = [{
        'type': 'multiline',
        'x': 10,
        'y': 10,
        'value': 'Line 1|Line 2|Line 3',
        'size': 18,
        'color': 'black',
        'delimiter': '|',
        'offset_y': 25
    }]

    result = customimage('entity_id', mock_service, mock_hass)
    generated_img = Image.open(io.BytesIO(result))
    example_img = Image.open(os.path.join(BASE_IMG_PATH, 'text_multiline_delimiter.png'))

    assert images_equal(generated_img, example_img), "Multiline text with custom delimiter failed"

def test_multiline_with_empty_lines(mock_hass, mock_service):
    """Test multiline text with empty lines."""
    mock_service.data['payload'] = [{
        'type': 'multiline',
        'x': 10,
        'y': 10,
        'value': 'Line 1||Line 3',
        'size': 18,
        'color': 'black',
        'delimiter': '|',
        'offset_y': 25
    }]
    result = customimage('entity_id', mock_service, mock_hass)
    generated_img = Image.open(io.BytesIO(result))
    example_img = Image.open(os.path.join(BASE_IMG_PATH, 'multiline_empty_line.png'))

    assert images_equal(generated_img, example_img), "Multiline text with empty lines failed"

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

def test_rectangle_rounded_corners(mock_hass, mock_service):
    """
    Test drawing a rectangle with rounded corners.
    """
    mock_service.data['payload'] = [{
        'type': 'rectangle',
        'x_start': 50,
        'y_start': 20,
        'x_end': 200,
        'y_end': 100,
        'fill': 'green',
        'outline': 'black',
        'width': 2,
        'radius': 15,
        'corners': 'top_left,bottom_right'
    }]

    result = customimage('entity_id', mock_service, mock_hass)
    generated_img = Image.open(io.BytesIO(result))
    example_img = Image.open(os.path.join(BASE_IMG_PATH, 'rectangle_rounded_corners.png'))

    assert images_equal(generated_img, example_img), "Rectangle with rounded corners failed"

# Rectangle pattern

def test_rectangle_pattern(mock_hass, mock_service):
    """
    Test drawing a pattern of rectangles.
    """
    mock_service.data['payload'] = [{
        'type': 'rectangle_pattern',
        'x_start': 10,
        'y_start': 10,
        'x_size': 30,
        'y_size': 30,
        'x_repeat': 5,
        'y_repeat': 3,
        'x_offset': 10,
        'y_offset': 10,
        'fill': 'red',
        'outline': 'black',
        'width': 1
    }]

    result = customimage('entity_id', mock_service, mock_hass)
    generated_img = Image.open(io.BytesIO(result))
    example_img = Image.open(os.path.join(BASE_IMG_PATH, 'rectangle_pattern.png'))

    assert images_equal(generated_img, example_img), "Rectangle pattern failed"

# Icon

# def test_icon_drawing(mock_hass, mock_service):
#     """
#     Test drawing an icon using MaterialDesign icons.
#     """
#     mock_service.data['payload'] = [{
#         'type': 'icon',
#         'x': 148,
#         'y': 64,
#         'value': 'mdi:home',
#         'size': 48,
#         'color': 'blue'
#     }]
#
#     result = customimage('entity_id', mock_service, mock_hass)
#     # TODO fix
#     generated_img = Image.open(io.BytesIO(result))
#     example_img = Image.open(os.path.join(BASE_IMG_PATH, 'icon_drawing.png'))
#
#     assert images_equal(generated_img, example_img), "Icon drawing failed"

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

# Ellipse

def test_ellipse_drawing(mock_hass, mock_service):
    """
    Test drawing an ellipse.
    """
    mock_service.data['payload'] = [{
        'type': 'ellipse',
        'x_start': 50,
        'y_start': 20,
        'x_end': 200,
        'y_end': 100,
        'fill': 'red',
        'outline': 'black',
        'width': 2
    }]

    result = customimage('entity_id', mock_service, mock_hass)
    generated_img = Image.open(io.BytesIO(result))
    example_img = Image.open(os.path.join(BASE_IMG_PATH, 'ellipse_drawing.png'))

    assert images_equal(generated_img, example_img), "Ellipse drawing failed"

# QR code

def test_qr_code(mock_hass, mock_service):
    """
    Test generating and drawing a QR code.
    """
    mock_service.data['payload'] = [{
        'type': 'qrcode',
        'x': 5,
        'y': 10,
        'data': 'https://www.youtube.com/watch?v=dQw4w9WgXcQ',
        'color': 'black',
        'bgcolor': 'white',
        'boxsize': 3,
        'border': 1
    }]
    # TODO border?
    result = customimage('entity_id', mock_service, mock_hass)
    generated_img = Image.open(io.BytesIO(result))
    example_img = Image.open(os.path.join(BASE_IMG_PATH, 'qr_code.png'))

    assert images_equal(generated_img, example_img), "QR code generation failed"

# def test_qrcode_with_empty_data(mock_hass, mock_service):
#     """Test QR code with empty data."""
#     mock_service.data['payload'] = [{
#         'type': 'qrcode',
#         'x': 10,
#         'y': 10,
#         'data': '',
#         'color': 'black',
#         'bgcolor': 'white',
#         'boxsize': 3,
#         'border': 4
#     }]
#     # with pytest.raises(ValueError):
#     result = customimage('entity_id', mock_service, mock_hass)
#     save_image(result)

def test_qrcode_with_very_long_data(mock_hass, mock_service):
    """Test QR code with very long data."""
    long_data = "https://example.com/" + "a" * 1000
    mock_service.data['payload'] = [{
        'type': 'qrcode',
        'x': 10,
        'y': 10,
        'data': long_data,
        'color': 'black',
        'bgcolor': 'white',
        'boxsize': 3,
        'border': 4
    }]
    result = customimage('entity_id', mock_service, mock_hass)
    generated_img = Image.open(io.BytesIO(result))
    example_img = Image.open(os.path.join(BASE_IMG_PATH, 'qr_code_long.png'))

    assert images_equal(generated_img, example_img), "Long QR code generation failed"

# Progress bar

def test_progress_bar(mock_hass, mock_service):
    """
    Test drawing a progress bar.
    """
    mock_service.data['payload'] = [{
        'type': 'progress_bar',
        'x_start': 10,
        'y_start': 50,
        'x_end': 286,
        'y_end': 70,
        'progress': 75,
        'fill': 'red',
        'outline': 'black',
        'width': 1,
        'show_percentage': True
    }]

    result = customimage('entity_id', mock_service, mock_hass)
    generated_img = Image.open(io.BytesIO(result))
    example_img = Image.open(os.path.join(BASE_IMG_PATH, 'progress_bar.png'))

    assert images_equal(generated_img, example_img), "Progress bar drawing failed"

def test_progress_bar_with_zero_progress(mock_hass, mock_service):
    """Test progress bar with zero progress."""
    mock_service.data['payload'] = [{
        'type': 'progress_bar',
        'x_start': 10,
        'y_start': 50,
        'x_end': 286,
        'y_end': 70,
        'progress': 0,
        'fill': 'red',
        'background': 'white',
        'outline': 'black',
        'width': 1,
        'show_percentage': True
    }]
    result = customimage('entity_id', mock_service, mock_hass)
    generated_img = Image.open(io.BytesIO(result))
    example_img = Image.open(os.path.join(BASE_IMG_PATH, 'progress_bar_zero.png'))

    assert images_equal(generated_img, example_img), "Progress bar zero progress failed"


def test_progress_bar_with_full_progress(mock_hass, mock_service):
    """Test progress bar with 100% progress."""
    mock_service.data['payload'] = [{
        'type': 'progress_bar',
        'x_start': 10,
        'y_start': 50,
        'x_end': 286,
        'y_end': 70,
        'progress': 100,
        'fill': 'red',
        'background': 'white',
        'outline': 'black',
        'width': 1,
        'show_percentage': True
    }]
    result = customimage('entity_id', mock_service, mock_hass)
    generated_img = Image.open(io.BytesIO(result))
    example_img = Image.open(os.path.join(BASE_IMG_PATH, 'progress_bar_full.png'))

    assert images_equal(generated_img, example_img), "Progress bar full progress failed"

# Plot
#
# from datetime import datetime, timedelta
#
# def test_plot(mock_hass, mock_service):
#     """
#     Test drawing a simple plot.
#     """
#     mock_service.data['payload'] = [{
#         'type': 'plot',
#         'data': [
#             {'entity': 'sensor.temperature', 'color': 'red'},
#             {'entity': 'sensor.humidity', 'color': 'blue'}
#         ],
#         'duration': 86400,  # 1 day
#         'low': 0,
#         'high': 100
#     }]
#
#     # Mock the get_significant_states function
#     with patch('custom_components.open_epaper_link.imagegen.get_significant_states') as mock_get_states:
#         now = datetime.now()
#         mock_get_states.return_value = {
#             'sensor.temperature': [
#                 MagicMock(state='20', last_changed=now - timedelta(hours=24)),
#                 MagicMock(state='22', last_changed=now - timedelta(hours=12)),
#                 MagicMock(state='21', last_changed=now)
#             ],
#             'sensor.humidity': [
#                 MagicMock(state='50', last_changed=now - timedelta(hours=24)),
#                 MagicMock(state='55', last_changed=now - timedelta(hours=12)),
#                 MagicMock(state='52', last_changed=now)
#             ]
#         }
#
#         result = customimage('entity_id', mock_service, mock_hass)
#         save_image(result)
#         generated_img = Image.open(io.BytesIO(result))
#         example_img = Image.open(os.path.join(BASE_IMG_PATH, 'plot.png'))
#
#         assert images_equal(generated_img, example_img), "Plot drawing failed"

def test_invalid_entity_for_plot(mock_hass, mock_service):
    """Test handling of invalid entity for plot."""

    mock_service.data['payload'] = [{
        'type': 'plot',
        'data': [{'entity': 'nonexistent.entity', 'color': 'red'}],
        'duration': 86400,
    }]

    with patch('custom_components.open_epaper_link.imagegen.get_significant_states', return_value={}):
        with pytest.raises(HomeAssistantError):
            customimage('entity_id', mock_service, mock_hass)

# def test_malformed_json_in_plot(mock_hass, mock_service):
#     """Test handling of malformed JSON in plot data."""
#     mock_service.data['payload'] = [{
#         'type': 'plot',
#         'data': 'not_a_valid_json',
#         'duration': 86400,
#     }]
#     with pytest.raises(TypeError):
#         customimage('entity_id', mock_service, mock_hass)
#
# def test_empty_plot_data(mock_hass, mock_service):
#     """Test handling of empty plot data."""
#     mock_service.data['payload'] = [{
#         'type': 'plot',
#         'data': [],
#         'duration': 86400,
#     }]
#     result = customimage('entity_id', mock_service, mock_hass)

# Image

# def test_invalid_image_url(mock_hass, mock_service):
#     """Test handling of invalid image URL."""
#     mock_service.data['payload'] = [
#         {'type': 'dlimg', 'x': 0, 'y': 0, 'url': 'not_a_valid_url', 'xsize': 100, 'ysize': 100}
#     ]
#     with pytest.raises(ValueError):
#         customimage('entity_id', mock_service, mock_hass)

def test_unsupported_image_format(mock_hass, mock_service):
    """Test handling of unsupported image format."""
    with patch('requests.get') as mock_get:
        mock_response = MagicMock()
        mock_response.content = b'Not an image'
        mock_get.return_value = mock_response

        mock_service.data['payload'] = [
            {'type': 'dlimg', 'x': 0, 'y': 0, 'url': 'http://example.com/image.xyz', 'xsize': 100, 'ysize': 100}
        ]

        with pytest.raises(IOError):
            customimage('entity_id', mock_service, mock_hass)

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

# def test_invalid_rotation_value(mock_hass, mock_service):
#     """Test handling of invalid rotation value."""
#
#     mock_service.data['rotate'] = 45  # Invalid rotation value
#     mock_service.data['payload'] = [{'type': 'text', 'x': 0, 'y': 0, 'value': 'Test'}]
#     # TODO implement error
#     with pytest.raises(ValueError, match="Rotation must be 0, 90, 180, or 270 degrees"):
#         result = customimage('entity_id', mock_service, mock_hass)
#         save_image(result)

def test_invalid_color_values(mock_hass, mock_service):
    """Test handling of invalid color values."""

    mock_service.data['payload'] = [
        {'type': 'text', 'x': 0, 'y': 0, 'value': 'Test', 'color': 'invalid_color'},
        {'type': 'line', 'x_start': 0, 'y_start': 0, 'x_end': 100, 'y_end': 100, 'fill': 'invalid_color'},
    ]

    result = customimage('entity_id', mock_service, mock_hass)
    generated_img = Image.open(io.BytesIO(result))

    # Assert that the image was created (invalid colors should default to white)
    assert generated_img.mode == 'RGB'
    # Assert that the image is white
    example_img = Image.open(os.path.join(BASE_IMG_PATH, 'blank.png'))
    assert images_equal(generated_img, example_img)

def test_empty_payload(mock_hass, mock_service):
    """Test handling of empty payload."""

    mock_service.data['payload'] = []

    result = customimage('entity_id', mock_service, mock_hass)
    generated_img = Image.open(io.BytesIO(result))

    # Assert that a blank image was created
    assert generated_img.mode == 'RGB'
    assert generated_img.size == (296, 128)  # Assuming default size

# def test_unknown_element_type(mock_hass, mock_service):
#     """Test handling of unknown element type."""
#
#     mock_service.data['payload'] = [{'type': 'unknown_type', 'x': 0, 'y': 0}]
#     TODO implement error
#     with pytest.raises(ValueError, match="Unknown element type: unknown_type"):
#         customimage('entity_id', mock_service, mock_hass)

def test_oversized_elements(mock_hass, mock_service):
    """Test handling of elements that exceed image boundaries."""

    mock_service.data['payload'] = [
        {'type': 'rectangle', 'x_start': 10, 'y_start': 0, 'x_end': 1000, 'y_end': 20, 'fill': 'red'},
        {'type': 'circle', 'x': 300, 'y': 100, 'radius': 70, 'fill': 'black'},
    ]
    result = customimage('entity_id', mock_service, mock_hass)
    generated_img = Image.open(io.BytesIO(result))
    example_img = Image.open(os.path.join(BASE_IMG_PATH, 'oversized_elements.png'))

    # Assert that the image was created (oversized elements should be clipped)
    assert generated_img.mode == 'RGB'
    assert generated_img.size == (296, 128)  # Assuming default size
    assert images_equal(generated_img, example_img)

def test_multiple_overlapping_elements(mock_hass, mock_service):
    """Test rendering multiple overlapping elements."""
    mock_service.data['payload'] = [
        {'type': 'rectangle', 'x_start': 0, 'y_start': 0, 'x_end': 100, 'y_end': 100, 'fill': 'red'},
        {'type': 'circle', 'x': 50, 'y': 50, 'radius': 30, 'fill': 'blue'},
        {'type': 'text', 'x': 20, 'y': 20, 'value': 'Overlapping', 'size': 20}
    ]
    result = customimage('entity_id', mock_service, mock_hass)
    generated_img = Image.open(io.BytesIO(result))
    example_img = Image.open(os.path.join(BASE_IMG_PATH, 'overlapping_elements.png'))
    assert images_equal(generated_img, example_img)

def test_all_element_types_together(mock_hass, mock_service):
    """Test rendering all supported element types in one image."""
    mock_service.data['payload'] = [
        {'type': 'rectangle', 'x_start': 0, 'y_start': 0, 'x_end': 100, 'y_end': 100, 'fill': 'black'},
        {'type': 'circle', 'x': 150, 'y': 50, 'radius': 30, 'fill': 'blue'},
        {'type': 'line', 'x_start': 0, 'y_start': 0, 'x_end': 296, 'y_end': 128, 'fill': 'black'},
        {'type': 'text', 'x': 10, 'y': 110, 'value': 'All Types', 'size': 20},
        {'type': 'ellipse', 'x_start': 200, 'y_start': 10, 'x_end': 280, 'y_end': 50, 'fill': 'red'},
        # {'type': 'icon', 'x': 250, 'y': 100, 'value': 'mdi:home', 'size': 24, 'color': 'black'},
        {'type': 'qrcode', 'x': 220, 'y': 60, 'data': 'https://example.com', 'boxsize': 2},
        {'type': 'progress_bar', 'x_start': 10, 'y_start': 80, 'x_end': 200, 'y_end': 100, 'progress': 75}
    ]
    result = customimage('entity_id', mock_service, mock_hass)
    generated_img = Image.open(io.BytesIO(result))
    example_img = Image.open(os.path.join(BASE_IMG_PATH, 'all_elements.png'))
    assert images_equal(generated_img, example_img)

# def test_zero_size_elements(mock_hass, mock_service):
#     """Test handling of elements with zero size."""
#     mock_service.data['payload'] = [
#         {'type': 'rectangle', 'x_start': 0, 'y_start': 0, 'x_end': 0, 'y_end': 0, 'fill': 'red'},
#         {'type': 'circle', 'x': 50, 'y': 50, 'radius': 0, 'fill': 'black'},
#         {'type': 'line', 'x_start': 0, 'y_start': 0, 'x_end': 0, 'y_end': 0, 'fill': 'red'}
#     ]
#     result = customimage('entity_id', mock_service, mock_hass)
#     TODO fix
#     generated_img = Image.open(io.BytesIO(result))
#     example_img = Image.open(os.path.join(BASE_IMG_PATH, 'blank.png'))
#     assert images_equal(generated_img, example_img)

def test_negative_coordinates(mock_hass, mock_service):
    """Test handling of elements with negative coordinates."""
    mock_service.data['payload'] = [
        {'type': 'rectangle', 'x_start': -10, 'y_start': -10, 'x_end': 50, 'y_end': 50, 'fill': 'red'},
        {'type': 'text', 'x': -20, 'y': -5, 'value': 'Negative', 'size': 20}
    ]
    result = customimage('entity_id', mock_service, mock_hass)
    generated_img = Image.open(io.BytesIO(result))
    example_img = Image.open(os.path.join(BASE_IMG_PATH, 'negative_coordinates.png'))
    assert images_equal(generated_img, example_img)