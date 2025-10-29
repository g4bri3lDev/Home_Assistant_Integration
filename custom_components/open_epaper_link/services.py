from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from functools import wraps
from typing import Final, Callable, List, Any

import async_timeout
import requests

from requests_toolbelt import MultipartEncoder

from homeassistant.core import HomeAssistant, ServiceCall
from homeassistant.exceptions import ServiceValidationError
from homeassistant.helpers import device_registry as dr
from homeassistant.helpers.dispatcher import async_dispatcher_send
from .const import DOMAIN, SIGNAL_TAG_IMAGE_UPDATE
from .imagegen import ImageGen
from .tag_types import get_tag_types_manager
from .util import send_tag_cmd, reboot_ap, is_ble_entry, get_ap_coordinator_from_hass
from .ble_utils import upload_image as ble_upload_image, DeviceMetadata

_LOGGER: Final = logging.getLogger(__name__)

DITHER_DISABLED = 0
DITHER_FLOYD_STEINBERG = 1
DITHER_ORDERED = 2
DITHER_DEFAULT = DITHER_ORDERED

MAX_RETRIES = 3
INITIAL_BACKOFF = 2  # seconds


def rgb_to_rgb332(rgb):
    """Convert RGB values to RGB332 format.

    Converts a standard RGB color tuple (0-255 for each component)
    to the 8-bit RGB332 format used by OpenEPaperLink for LED patterns.

    The conversion uses:

    - 3 bits for red (0-7)
    - 3 bits for green (0-7)
    - 2 bits for blue (0-3)

    Args:
        rgb: Tuple of (r, g, b) values, each 0-255

    Returns:
        str: Hexadecimal string representation of the RGB332 value
    """
    r, g, b = [max(0, min(255, x)) for x in rgb]
    r = (r // 32) & 0b111
    g = (g // 32) & 0b111
    b = (b // 64) & 0b11
    rgb332 = (r << 5) | (g << 2) | b
    return str(hex(rgb332)[2:].zfill(2))


def int_to_hex_string(number: int) -> str:
    """Convert integer to two-digit hex string.

    Ensures the resulting hex string is always two digits,
    padding with a leading zero if needed.

    Args:
        number: Integer value to convert

    Returns:
        str: Two-digit hexadecimal string
    """
    hex_string = hex(number)[2:]
    return '0' + hex_string if len(hex_string) == 1 else hex_string


class UploadQueueHandler:
    """Handle queued image uploads to the AP.

    Manages a queue of image upload tasks to prevent overwhelming the AP with concurrent requests.

    Features include:

    - Maximum concurrent upload limit
    - Cooldown period between uploads
    - Task tracking and status reporting

    This helps maintain AP stability while processing multiple image requests from different parts of Home Assistant.
    """

    def __init__(self, max_concurrent: int = 1, cooldown: float = 1.0):
        """Initialize the upload queue handler.

        Args:
            max_concurrent: Maximum number of concurrent uploads (default: 1)
            cooldown: Cooldown period in seconds between uploads (default: 1.0)
        """
        self._queue = asyncio.Queue()
        self._processing = False
        self._max_concurrent = max_concurrent
        self._cooldown = cooldown
        self._active_uploads = 0
        self._last_upload = None
        self._lock = asyncio.Lock()

    def __str__(self):
        """Return queue status string."""
        return f"Queue(active={self._active_uploads}, size={self._queue.qsize()})"

    async def add_to_queue(self, upload_func, *args, **kwargs):
        """Add an upload task to the queue.

        Queues an upload function with its arguments for later execution.
        Starts the queue processor if it's not already running.

        Args:
            upload_func: Async function that performs the actual upload
            *args: Positional arguments to pass to the upload function
            **kwargs: Keyword arguments to pass to the upload function
        """

        entity_id = next((arg for arg in args if isinstance(arg, str) and "." in arg), "unknown")

        _LOGGER.debug("Adding upload task to queue for %s. %s", entity_id, self)
        # Add task to queue
        await self._queue.put((upload_func, args, kwargs))

        # Start processing queue if not already running
        if not self._processing:
            _LOGGER.debug("Starting upload queue processor for %s", entity_id)
            asyncio.create_task(self._process_queue())

    async def _process_queue(self):
        """Process queued upload tasks with true parallelism.

        Long-running task that processes the upload queue, respecting:

        - Maximum concurrent upload limit
        - Cooldown period between uploads

        Creates background tasks for parallel execution instead of blocking.
        Handles errors in individual uploads without stopping queue processing.
        This method runs until the queue is empty, then terminates.
        """
        self._processing = True
        _LOGGER.debug("Upload queue processor started. %s", self)
        
        running_tasks = set()

        try:
            while not self._queue.empty() or running_tasks:
                # Clean up completed tasks
                if running_tasks:
                    done_tasks = {task for task in running_tasks if task.done()}
                    for task in done_tasks:
                        running_tasks.remove(task)
                        # Get the result to handle any exceptions
                        try:
                            await task
                        except Exception as err:
                            _LOGGER.error("Background upload task failed: %s", str(err))

                # Check if new uploads can be started
                async with self._lock:
                    if (not self._queue.empty() and 
                        self._active_uploads < self._max_concurrent):
                        
                        # Check cooldown period
                        if self._last_upload:
                            elapsed = (datetime.now() - self._last_upload).total_seconds()
                            if elapsed < self._cooldown:
                                _LOGGER.debug("In cooldown period (%.1f seconds remaining)",
                                              self._cooldown - elapsed)
                                await asyncio.sleep(self._cooldown - elapsed)

                        # Get next task from queue
                        upload_func, args, kwargs = await self._queue.get()
                        entity_id = next((arg for arg in args if isinstance(arg, str) and "." in arg), "unknown")

                        # Create and start background task
                        task = asyncio.create_task(self._execute_upload(upload_func, args, kwargs, entity_id))
                        running_tasks.add(task)
                        
                        # Update last upload timestamp
                        self._last_upload = datetime.now()
                        
                    else:
                        # Wait a bit before checking again
                        await asyncio.sleep(0.1)

        finally:
            # Wait for all running tasks to complete
            if running_tasks:
                await asyncio.gather(*running_tasks, return_exceptions=True)
            self._processing = False

    async def _execute_upload(self, upload_func, args, kwargs, entity_id):
        """Execute a single upload task in the background."""
        try:
            # Increment active uploads counter
            async with self._lock:
                self._active_uploads += 1
            _LOGGER.debug("Starting upload for %s. %s", entity_id, self)

            # Perform upload
            _LOGGER.debug("Starting queued upload task")
            start_time = datetime.now()
            await upload_func(*args, **kwargs)
            duration = (datetime.now() - start_time).total_seconds()

            _LOGGER.debug("Upload completed for %s in %.1f seconds", entity_id, duration)

        except Exception as err:
            _LOGGER.error("Error processing queued upload for %s: %s", entity_id, str(err))
        finally:
            # Decrement active upload counter
            async with self._lock:
                self._active_uploads -= 1
            # Mark task as done
            self._queue.task_done()
            _LOGGER.debug("Upload task for %s finished. %s", entity_id, self)


async def async_setup_services(hass: HomeAssistant) -> None:
    """Set up the OpenEPaperLink services.
    Args:
        hass: Home Assistant instance
    """

    # Separate queues for different device types
    ble_upload_queue = UploadQueueHandler(max_concurrent=1, cooldown=0.1)
    ap_upload_queue = UploadQueueHandler(max_concurrent=1, cooldown=1.0)

    def _extract_mac_from_entity_id(entity_id: str) -> str:
        """
        Extract the MAC address from an entity ID.

        Args:
            entity_id: Entity ID in format `domain.mac_address`

        Returns:
            str: Uppercase MAC address
        """
        return entity_id.split(".")[1].upper()

    def _get_device_by_entity_id(entity_id: str) -> dr.DeviceEntry | None:
        """
        Get device registry entry from entity_id.

        Args:
            entity_id: Entity ID in format `domain.mac_address`

        Returns:
            DeviceEntry if found, else None
        """
        mac = _extract_mac_from_entity_id(entity_id)
        device_registry = dr.async_get(hass)
        for device in device_registry.devices.values():
            for identifier in device.identifiers:
                if identifier[0] == DOMAIN:
                    device_mac = identifier[1]
                    if device_mac.startswith("ble_"):
                        device_mac = device_mac[4:]
                    if device_mac.upper() == mac:
                        return device

        return None

    def _is_ble_device(entity_id: str) -> bool:
        """
        Check if the entity represents a BLE device.

        Args:
            entity_id: Entity ID in format `domain.mac_address`

        Returns:
            bool: True if BLE device, False if APCoordinator device
        """
        device = _get_device_by_entity_id(entity_id)
        if not device:
            return False
        for identifier in device.identifiers:
            if identifier[0] == DOMAIN:
                if identifier[1].startswith("ble_"):
                    return True
        return False

    def _build_led_pattern(service_data: dict[str, Any]) -> str:
        """Build LED pattern hex string from service data.

        Creates the hex pattern string according to OpenEPaperLink LED control protocol.
        See: https://github.com/OpenEPaperLink/OpenEPaperLink/wiki/Led-control

        Args:
            service_data: Service call data containing LED parameters

        Returns:
            str: Hex pattern string for LED control
        """
        mode = service_data.get("mode", "")
        modebyte = "1" if mode == "flash" else "0"
        brightness = service_data.get("brightness", 2)
        modebyte = hex(((brightness - 1) << 4) + int(modebyte))[2:]

        def _color_segment(color_num: int) -> str:
            """Build a pattern segment for one color.

            Args:
                color_num: Color number (1, 2, or 3)

            Returns:
                str: Hex pattern segment for this color
            """
            default_delay = 0.0 if color_num == 3 else 0.1
            return (
                    rgb_to_rgb332(service_data.get(f"color{color_num}", "")) +
                    hex(int(service_data.get(f"flashSpeed{color_num}", 0.2) * 10))[2:] +
                    hex(service_data.get(f"flashCount{color_num}", 2))[2:] +
                    int_to_hex_string(int(service_data.get(f"delay{color_num}", default_delay) * 10))
            )

        return (
                modebyte +
                _color_segment(1) +
                _color_segment(2) +
                _color_segment(3) +
                int_to_hex_string(service_data.get("repeats", 2) - 1) +
                "00"
        )

    def require_ap_coordinator_online(func: Callable) -> Callable:
        """
        Decorator to require the AP to be online before executing a service.

        Args:
            func: Service handler function to wrap

        Returns:
            Wrapped service handler function that validates AP status before execution.
        """
        @wraps(func)
        async def wrapper(service: ServiceCall, *args, **kwargs) -> None:
            ap_coordinator = get_ap_coordinator_from_hass(hass)
            if not ap_coordinator.online:
                raise ServiceValidationError(
                    "OpenEPaperLink AP is offline. Please check your network connection and AP status."
                )
            return await func(service, *args, **kwargs)
        return wrapper

    def handle_targets(func: Callable) -> Callable:
        """
        Decorator to handle device_id, label_id, and area_id targeting.

        Resolves all three targeting methods to a list of entity_ids and calls the decorated function once per entity

        Args:
            func: Service handler function to wrap

        Returns:
            Wrapped service handler function that handles device_id, label_id, and area_id targeting.
        """
        # Helper functions specific to target resolution
        async def get_device_ids_from_label_id(label_id: str) -> List[str]:
            """
            Get device_ids for a label_id.

            Args:
                label_id: Home Assistant label ID

            Returns:
                List of device IDs
            """
            device_registry = dr.async_get(hass)
            devices = dr.async_entries_for_label(device_registry, label_id)
            return [device.id for device in devices]

        async def get_device_ids_from_area_id(area_id: str) -> List[str]:
            """
            Get device_ids for all OpenEPaperLink devices in an area.

            Args:
                area_id: Home Assistant area ID

            Returns:
                List of device IDs for OpenEPaperLink devices only
            """
            device_registry = dr.async_get(hass)
            devices = dr.async_entries_for_area(device_registry, area_id)

            oepl_device_ids = []
            for device in devices:
                for identifier in device.identifiers:
                    if identifier[0] == DOMAIN:
                        oepl_device_ids.append(device.id)
                        break

            return oepl_device_ids

        async def get_entity_id_from_device_id(device_id: str) -> str:
            """Get the primary entity ID for an OpenEPaperLink device.

            Args:
                device_id: Home Assistant device ID

            Returns:
                str: Entity ID in format "open_epaper_link.mac_address"

            Raises:
                ServiceValidationError: If a device is not found or an OpenEPaperLink device
            """
            device_registry = dr.async_get(hass)
            device = device_registry.async_get(device_id)

            if not device:
                raise ServiceValidationError(f"Device {device_id} not found")

            if not device.identifiers:
                raise ServiceValidationError(f"No identifiers found for device {device_id}")

            domain_mac = next(iter(device.identifiers))
            if domain_mac[0] != DOMAIN:
                raise ServiceValidationError(f"Device {device_id} is not an OpenEPaperLink device")

            identifier = domain_mac[1]
            if identifier.startswith("ble_"):
                mac_address = identifier[4:]
            else:
                mac_address = identifier

            return f"{DOMAIN}.{mac_address.lower()}"

        @wraps(func)
        async def wrapper(service: ServiceCall, *args, **kwargs):
            device_ids = service.data.get("device_id", [])
            label_ids = service.data.get("label_id", [])
            area_ids = service.data.get("area_id", [])

            # Normalize to lists
            if isinstance(device_ids, str):
                device_ids = [device_ids]
            if isinstance(label_ids, str):
                label_ids = [label_ids]
            if isinstance(area_ids, str):
                area_ids = [area_ids]

            # Expand labels
            for label_id in label_ids:
                expanded = await get_device_ids_from_label_id(label_id)
                device_ids.extend(expanded)

            # Expand areas
            for area_id in area_ids:
                expanded = await get_device_ids_from_area_id(area_id)
                device_ids.extend(expanded)

            # Remove duplicates
            seen = set()
            unique_device_ids = []
            for device_id in device_ids:
                if device_id not in seen:
                    seen.add(device_id)
                    unique_device_ids.append(device_id)

            if not unique_device_ids:
                raise ServiceValidationError(
                    "No target devices specified. Please provide device_id, label_id, or area_id."
                )

                # Process each device
            errors = []
            for device_id in unique_device_ids:
                try:
                    entity_id = await get_entity_id_from_device_id(device_id)
                    await func(service, entity_id, *args, **kwargs)
                except ServiceValidationError as err:
                    error_msg = f"Error processing device {device_id}: {err}"
                    _LOGGER.error(error_msg)
                    errors.append(error_msg)
                    continue

            # If all devices failed, raise a combined error
            if errors and len(errors) == len(unique_device_ids):
                raise ServiceValidationError("\n".join(errors))
        return wrapper

    @handle_targets
    async def drawcustom_service(service: ServiceCall, entity_id: str) -> None:
        """
        Handle drawcustom service calls.

        Processes requests to generate and upload custom images to tags.
        The service supports:

        - Multiple target devices
        - Custom content with text, shapes, and images
        - Background color and rotation
        - Dithering options
        - "Dry run" mode for testing

        Args:
            service: Service call object with parameters
            entity_id: Target entity ID (resolved by decorator)

        Raises:
            ServiceValidationError: If AP is offline or image generation fails
        """
        device_errors = []

        try:
            # Determine if BLE device or APCoordinator device
            is_ble_device = _is_ble_device(entity_id)

            # For APCoordinator devices, ensure APCoordinator is online
            ap_coordinator = None
            if not is_ble_device:
                ap_coordinator = get_ap_coordinator_from_hass(hass)
                if not ap_coordinator.online:
                    raise ServiceValidationError(
                        "OpenEPaperLink AP is offline. Please check your network connection and AP status."
                    )

            # Generate image
            generator = ImageGen(hass)

            if is_ble_device:
                tag_info = await generator.get_ble_tag_info(hass, entity_id)
                image_data = await generator.generate_custom_image(
                    entity_id=entity_id,
                    service_data=service.data,
                    error_collector=device_errors,
                    tag_info=tag_info
                )
            else:
                image_data = await generator.generate_custom_image(
                    entity_id=entity_id,
                    service_data=service.data,
                    error_collector=device_errors
                )

            if device_errors:
                _LOGGER.warning(
                    "Completed with warnings for device %s:\n%s",
                    entity_id,
                    "\n".join(device_errors)
                )

            # Handle dry-run mode
            if service.data.get("dry-run", False):
                _LOGGER.info("Dry run completed for %s", entity_id)
                tag_mac = entity_id.split(".")[1].upper()
                async_dispatcher_send(
                    hass,
                    f"{SIGNAL_TAG_IMAGE_UPDATE}_{tag_mac}",
                    image_data
                )
                return

            # Upload image
            if is_ble_device:
                from .util import is_bluetooth_available
                if not is_bluetooth_available(hass):
                    raise ServiceValidationError(
                        f"Cannot upload to BLE device {entity_id}: "
                        "Bluetooth integration is disabled or no scanners available. "
                        "Please enable Bluetooth integration in Home Assistant."
                    )

                await ble_upload_queue.add_to_queue(
                    upload_ble_image,
                    entity_id,
                    image_data
                )
            else:
                await ap_upload_queue.add_to_queue(
                    upload_image,
                    ap_coordinator,
                    entity_id,
                    image_data,
                    service.data.get("dither", DITHER_DEFAULT),
                    service.data.get("ttl", 60),
                    service.data.get("preload_type", 0),
                    service.data.get("preload_lut", 0)
                )

        except ServiceValidationError:
            raise
        except Exception as err:
            raise ServiceValidationError(f"Error processing device {entity_id}: {str(err)}") from err

    async def upload_image(ap_coordinator, entity_id: str, img: bytes, dither: int, ttl: int,
                           preload_type: int = 0, preload_lut: int = 0) -> None:
        """
        Upload image to tag through AP.

        Sends an image to the AP for display on a specific tag using
        multipart/form-data POST request. Configures display parameters
        such as dithering, TTL, and optional preloading.

        Will retry upload on timeout, with exponential backoff.

        Args:
            ap_coordinator: APCoordinator instance with connection details
            entity_id: Entity ID of the target tag
            img: JPEG image data as bytes
            dither: Dithering mode (0=none, 1=Floyd-Steinberg, 2=ordered)
            ttl: Time-to-live in seconds
            preload_type: Type for image preloading (0=disabled)
            preload_lut: Look-up table for preloading

        Raises:
            ServiceValidationError: If upload fails or times out
        """
        url = f"http://{ap_coordinator.host}/imgupload"
        mac = _extract_mac_from_entity_id(entity_id)

        _LOGGER.debug("Upload parameters: dither=%d, ttl=%d, preload_type=%d, preload_lut=%d",
                      dither, ttl, preload_type, preload_lut)

        ttl_minutes = max(1, ttl // 60)
        backoff_delay = INITIAL_BACKOFF

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                fields = {
                    'mac': mac,
                    'contentmode': "25",
                    'dither': str(dither),
                    'ttl': str(ttl_minutes),
                    'image': ('image.jpg', img, 'image/jpeg'),
                }

                if preload_type > 0:
                    fields.update({
                        'preloadtype': str(preload_type),
                        'preloadlut': str(preload_lut),
                    })

                mp_encoder = MultipartEncoder(fields=fields)

                async with async_timeout.timeout(30):
                    response = await hass.async_add_executor_job(
                        lambda: requests.post(
                            url,
                            headers={'Content-Type': mp_encoder.content_type},
                            data=mp_encoder
                        )
                    )

                if response.status_code != 200:
                    raise ServiceValidationError(
                        f"Image upload failed for {entity_id} with status code: {response.status_code}"
                    )
                break

            except asyncio.TimeoutError:
                if attempt < MAX_RETRIES:
                    _LOGGER.warning(
                        "Timeout uploading %s (attempt %d/%d), retrying in %ds…",
                        entity_id, attempt, MAX_RETRIES, backoff_delay
                    )
                    await asyncio.sleep(backoff_delay)
                    backoff_delay *= 2
                    continue
                raise ServiceValidationError(f"Image upload timed out for {entity_id}")
            except Exception as err:
                raise ServiceValidationError(f"Failed to upload image for {entity_id}: {str(err)}") from err

    async def upload_ble_image(entity_id: str, img: bytes) -> None:
        """
        Upload image to BLE tag.

        Sends an image to a BLE tag using direct Bluetooth communication.
        This bypasses the AP and provides faster upload times.

        Args:
            entity_id: Entity ID of the target tag
            img: JPEG image data as bytes

        Raises:
            ServiceValidationError: If BLE upload fails
        """
        mac = _extract_mac_from_entity_id(entity_id)

        try:
            domain_data = hass.data.get(DOMAIN, {})
            device_metadata = None

            for entry_id, entry_data in domain_data.items():
                if (is_ble_entry(entry_data) and
                        entry_data.get("mac_address", "").upper() == mac):
                    device_metadata = entry_data.get("device_metadata", {})
                    break

            if not device_metadata:
                raise ServiceValidationError(f"No metadata found for BLE device {entity_id}")

            metadata = DeviceMetadata(
                hw_type=device_metadata.get("hw_type", 0),
                fw_version=device_metadata.get("fw_version", 0),
                width=device_metadata.get("width", 0),
                height=device_metadata.get("height", 0),
                color_support=device_metadata.get("color_support", "mono"),
                rotatebuffer=device_metadata.get("rotatebuffer", 0)
            )

            success = await ble_upload_image(hass, mac, img, metadata)
            if not success:
                raise ServiceValidationError(f"BLE image upload failed for {entity_id}")

        except ServiceValidationError:
            raise
        except Exception as err:
            raise ServiceValidationError(f"Failed to upload image via BLE to {entity_id}: {str(err)}") from err

    @require_ap_coordinator_online
    @handle_targets
    async def setled_service(service: ServiceCall, entity_id: str) -> None:
        """
        Handle LED pattern service calls.

        Configures LED flashing patterns for tags. Supports:

        - Off/flashing mode
        - Brightness settings
        - Multi-color patterns with timing
        - Repeat counts

        The LED pattern is encoded as a hex string according to the
        OpenEPaperLink protocol specification.

        Args:
            service: Service call object with parameters
            entity_id: Target entity ID (resolved by decorator)

        Raises:
            ServiceValidationError: If request fails
        """
        ap_coordinator = get_ap_coordinator_from_hass(hass)
        mac = _extract_mac_from_entity_id(entity_id)
        pattern = _build_led_pattern(service.data)

        url = f"http://{ap_coordinator.host}/led_flash?mac={mac}&pattern={pattern}"
        result = await hass.async_add_executor_job(requests.get, url)

        if result.status_code != 200:
            raise ServiceValidationError(
                f"LED pattern update failed with status code: {result.status_code}"
            )

    @require_ap_coordinator_online
    @handle_targets
    async def clear_pending_service(service: ServiceCall, entity_id: str) -> None:
        """
        Clear pending updates for target devices.

        Sends command to clear any pending updates for the target tags,
        canceling queued content changes that haven't been applied yet.

        Args:
            service: Service call object
            entity_id: Target entity ID (resolved by decorator)
        """
        await send_tag_cmd(hass, entity_id, "clear")

    @require_ap_coordinator_online
    @handle_targets
    async def force_refresh_service(service: ServiceCall, entity_id: str) -> None:
        """
        Force refresh target devices.

        Sends command to force the refresh of the tag display,
        to, for example, redraw content.

        Args:
            service: Service call object
            entity_id: Target entity ID (resolved by decorator)
        """
        await send_tag_cmd(hass, entity_id, "refresh")

    @require_ap_coordinator_online
    @handle_targets
    async def reboot_tag_service(service: ServiceCall, entity_id: str) -> None:
        """
        Reboot target devices.

        Sends command to reboot the target tags, performing a full
        restart of the tags.

        Args:
            service: Service call object
            entity_id: Target entity ID (resolved by decorator)
        """
        await send_tag_cmd(hass, entity_id, "reboot")

    async def scan_channels_service(service: ServiceCall, entity_id: str) -> None:
        """
        Trigger channel scan on target devices.

        Sends command to trigger an IEEE 802.15.4 channel scan on the
        target tags.

        Args:
            service: Service call object
            entity_id: Target entity ID (resolved by decorator)
        """
        await send_tag_cmd(hass, entity_id, "scan")

    @require_ap_coordinator_online
    async def reboot_ap_service(service: ServiceCall) -> None:
        """
        Reboot the Access Point.

        Sends command to reboot the Access Point, performing a full
        restart of the AP firmware. This temporarily disconnects all tags.

        Args:
            service: Service call object
        """
        await reboot_ap(hass)

    async def refresh_tag_types_service(service: ServiceCall) -> None:
        """
        Refresh tag type definitions from GitHub.

        Forces a refresh of tag type definitions from the GitHub repository,
        updating the local cache with the latest hardware support information.

        Creates a persistent notification when complete to inform the user.

        Args:
            service: Service call object
        """
        manager = await get_tag_types_manager(hass)
        manager._last_update = None  # Force refresh
        await manager.ensure_types_loaded()

        tag_types_len = len(manager.get_all_types())
        message = f"Successfully refreshed {tag_types_len} tag types from GitHub"

        await hass.services.async_call(
            "persistent_notification",
            "create",
            {
                "title": "Tag Types Refreshed",
                "message": message,
                "notification_id": "tag_types_refresh_notification",
            },
        )
    # Register services available for all device types
    hass.services.async_register(DOMAIN, "drawcustom", drawcustom_service)
    hass.services.async_register(DOMAIN, "setled", setled_service)
    hass.services.async_register(DOMAIN, "clear_pending", clear_pending_service)
    hass.services.async_register(DOMAIN, "force_refresh", force_refresh_service)
    hass.services.async_register(DOMAIN, "reboot_tag", reboot_tag_service)
    hass.services.async_register(DOMAIN, "scan_channels", scan_channels_service)
    hass.services.async_register(DOMAIN, "reboot_ap", reboot_ap_service)
    hass.services.async_register(DOMAIN, "refresh_tag_types", refresh_tag_types_service)