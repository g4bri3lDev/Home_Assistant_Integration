import os
import pytest
from unittest.mock import patch, MagicMock
from PIL import Image, ImageChops
import io
from custom_components.open_epaper_link.imagegen import customimage, getIndexColor
BASE_IMG_PATH = 'tests/components/open_epaper_link/test_images'
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

