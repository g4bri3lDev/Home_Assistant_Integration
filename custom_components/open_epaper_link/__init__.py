import logging
import os
from typing import Final, TypedDict

from homeassistant.components import persistent_notification
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import Platform, EVENT_HOMEASSISTANT_STARTED, CONF_HOST
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import ConfigEntryNotReady, ConfigEntryError
from homeassistant.helpers import entity_registry as er, device_registry as dr
from homeassistant.helpers import issue_registry as ir
from homeassistant.helpers.typing import ConfigType
from .ble_utils import interrogate_ble_device, BLETimeoutError, BLEConnectionError, BLEError
from .const import DOMAIN
from .hub import Hub
from .services import async_setup_services
from .util import is_ble_entry

_LOGGER: Final = logging.getLogger(__name__)

PLATFORMS = [
    Platform.SENSOR,
    Platform.BUTTON,
    Platform.IMAGE,
    Platform.SELECT,
    Platform.SWITCH,
    Platform.TEXT,
]

# BLE devices use a subset of platforms
BLE_PLATFORMS = [
    Platform.SENSOR,  # Battery, RSSI, last seen
    Platform.LIGHT,  # LED control
    Platform.BUTTON,  # Clock mode controls
]


class BLERuntimeData(TypedDict):
    """Runtime data for BLE devices."""
    type: str
    mac_address: str
    name: str
    device_metadata: dict
    sensors: dict


type OpenEPaperLinkConfigEntry = ConfigEntry[Hub | BLERuntimeData]


async def async_migrate_camera_entities(hass: HomeAssistant, entry: ConfigEntry) -> list[str]:
    """Migrate old camera entities to image entities.

          Finds and removes camera entities that match our unique ID pattern,
          returns a list of removed entity IDs for notification.

          Returns:
              list[str]: List of removed camera entity IDs
    """
    entity_registry = er.async_get(hass)
    removed_entities = []

    # Find camera entities with OEPL domain and content in unique_id
    camera_entities = []
    for entity in entity_registry.entities.values():
        if entity.platform == DOMAIN and entity.domain == "camera" and entity.unique_id.endswith("_content"):
            camera_entities.append(entity)

    for entity in camera_entities:
        _LOGGER.info("Removing old camera entity: %s", entity.entity_id)
        entity_registry.async_remove(entity.entity_id)
        removed_entities.append(entity.entity_id)

    return removed_entities


async def async_setup(hass: HomeAssistant, config: ConfigType) -> bool:
    """Set up the OpenEPaperLink integration."""
    await async_setup_services(hass)
    return True


async def async_setup_entry(hass: HomeAssistant, entry: OpenEPaperLinkConfigEntry) -> bool:
    """Set up OpenEPaperLink integration from a config entry.

    This is the main entry point for integration initialization, which handles both:
    - AP-based entries (traditional WebSocket-based integration)
    - BLE-based entries (direct Bluetooth communication)

    Args:
        hass: Home Assistant instance
        entry: Configuration Entry object with connection details

    Returns:
        bool: True if setup was successful, False otherwise
    """
    # Detect BLE vs AP entry type
    is_ble_device = entry.data.get("device_type") == "ble"

    if is_ble_device:
        # BLE device setup using a simple callback approach
        _LOGGER.debug("Setting up BLE device entry: %s", entry.data.get("name"))

        from homeassistant.components import bluetooth
        from .ble_utils import MANUFACTURER_ID, parse_ble_advertisement, calculate_battery_percentage
        from datetime import datetime, timezone

        mac_address = entry.data.get("mac_address")
        name = entry.data.get("name")
        device_metadata = entry.data.get("device_metadata", {})

        # Store BLE device config in hass.data for entity access
        ble_data = {
            "type": "ble",
            "mac_address": mac_address,
            "name": name,
            "device_metadata": device_metadata,
            "sensors": {},  # Registry of sensor entities
        }
        entry.runtime_data = ble_data

        # Validate BLE device is reachable before proceeding
        try:
            _LOGGER.debug("Performing initial BLE check for %s", mac_address)
            await interrogate_ble_device(hass, mac_address)
            _LOGGER.debug("BLE device %s is reachable", mac_address)
        except (BLEConnectionError, BLETimeoutError) as err:
            _LOGGER.warning("BLE device %s not reachable during setup: %s", mac_address, err)
            ir.async_create_issue(
                hass,
                DOMAIN,
                f"ble_not_reachable_{mac_address}",
                is_fixable=False,
                severity=ir.IssueSeverity.ERROR,
                translation_key="ble_not_reachable",
                translation_placeholders={"name": name, "mac_address": mac_address, "error": str(err)},
            )
            raise ConfigEntryNotReady(f"BLE device {name} ({mac_address}) not reachable") from err
        except BLEError as err:
            _LOGGER.error("BLE protocol error during setup for %s: %s", mac_address, err)
            ir.async_create_issue(
                hass,
                DOMAIN,
                f"ble_protocol_error_{mac_address}",
                is_fixable=False,
                severity=ir.IssueSeverity.ERROR,
                translation_key="ble_protocol_error",
                translation_placeholders={"name": name, "mac_address": mac_address, "error": str(err)},
            )
            raise ConfigEntryError(f"Failed to communicate with BLE device {name} ({mac_address})") from err
        except Exception as err:
            _LOGGER.error("Unexpected error during BLE device setup for %s: %s", mac_address, err)
            ir.async_create_issue(
                hass,
                DOMAIN,
                f"ble_setup_error_{mac_address}",
                is_fixable=False,
                severity=ir.IssueSeverity.ERROR,
                translation_key="ble_setup_error",
                translation_placeholders={"name": name, "mac_address": mac_address, "error": str(err)},
            )
            raise ConfigEntryError(f"Failed to setup BLE device {name} ({mac_address})") from err

        def _ble_device_found(
                service_info: bluetooth.BluetoothServiceInfoBleak,
                change: bluetooth.BluetoothChange,
        ) -> None:
            """Handle BLE advertising data updates."""

            # Only process the specific device
            if service_info.address != mac_address:
                return

            # Parse manufacturer data
            manufacturer_data = service_info.manufacturer_data.get(MANUFACTURER_ID)
            if not manufacturer_data:
                return

            parsed_data = parse_ble_advertisement(manufacturer_data)
            if not parsed_data:
                return

            # Dynamically update device attributes
            new_fw_version = parsed_data.get("fw_version")
            if new_fw_version is not None:
                device_registry = dr.async_get(hass)
                device_entry = device_registry.async_get_device(
                    identifiers={(DOMAIN, f"ble_{mac_address}")}
                )
                new_fw_string = str(new_fw_version)
                if device_entry and device_entry.sw_version != new_fw_string:
                    _LOGGER.debug(
                        "Device %s firmware updated from %s to %s",
                        mac_address,
                        device_entry.sw_version,
                        new_fw_string,
                    )
                    device_registry.async_update_device(
                        device_entry.id,
                        sw_version=new_fw_string
                    )

            # Build sensor data
            battery_mv = parsed_data.get("battery_mv", 0)
            sensor_data = {
                "battery_percentage": calculate_battery_percentage(battery_mv) if battery_mv > 0 else None,
                "battery_voltage": battery_mv if battery_mv > 0 else None,
                "rssi": service_info.rssi,
                "last_seen": datetime.now(timezone.utc),
                "temperature": parsed_data.get("temperature"),
            }

            # Update all registered sensors
            for sensor in ble_data["sensors"].values():
                sensor.update_from_advertising_data(sensor_data)

            _LOGGER.debug("Updated BLE sensors for %s: %s", mac_address, sensor_data)

        # Register BLE advertising listener
        unregister_callback = bluetooth.async_register_callback(
            hass,
            _ble_device_found,
            {"manufacturer_id": MANUFACTURER_ID},
            bluetooth.BluetoothScanningMode.ACTIVE,
        )
        entry.async_on_unload(unregister_callback)

        # Set up BLE-specific platforms
        await hass.config_entries.async_forward_entry_setups(entry, BLE_PLATFORMS)

        # Clear any previous setup failure repair issues
        ir.async_delete_issue(hass, DOMAIN, f"ble_not_reachable_{mac_address}")
        ir.async_delete_issue(hass, DOMAIN, f"ble_protocol_error_{mac_address}")
        ir.async_delete_issue(hass, DOMAIN, f"ble_setup_error_{mac_address}")

    else:
        # Traditional AP setup
        _LOGGER.debug("Setting up AP entry: %s", entry.data.get(CONF_HOST, "unknown"))

        hub = Hub(hass, entry)

        # Do basic setup without WebSocket connection
        # Will raise ConfigEntryNotReady or ConfigEntryError on failure
        await hub.async_setup_initial()

        entry.runtime_data = hub

        removed_entities = await async_migrate_camera_entities(hass, entry)
        if removed_entities:
            persistent_notification.async_create(
                hass,
                f"OpenEPaperLink: Migrated {len(removed_entities)} camera entities to image entities.\n\n"
                f"Please update your dashboards and automations to use the new image entities instead of camera entities.\n\n"
                f"Removed entities: {', '.join(removed_entities)}",
                title="OpenEPaperLink Migration",
                notification_id="open_epaper_link_camera_migration"
            )

        await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)

        async def start_websocket(_):
            """Start WebSocket connection after HA is fully started."""
            await hub.async_start_websocket()

        if hass.is_running:
            # If HA is already running, start WebSocket immediately
            await hub.async_start_websocket()
        else:
            # Otherwise wait for the started event
            hass.bus.async_listen_once(EVENT_HOMEASSISTANT_STARTED, start_websocket)

    # Listen for changes to options
    entry.async_on_unload(entry.add_update_listener(async_update_options))

    return True


async def async_update_options(hass: HomeAssistant, entry: OpenEPaperLinkConfigEntry) -> None:
    """Handle updates to integration options.

    Called when the user updates integration options through the UI.
    Only applies to AP-based entries (BLE devices don't have configurable options).

    Args:
        hass: Home Assistant instance
        entry: Updated configuration entry
    """
    # Only AP entries have hub with reload_config method
    if is_ble_entry(entry):
        # BLE devices don't have configurable options yet
        return

    # Traditional AP entry
    hub = entry.runtime_data
    await hub.async_reload_config()


async def async_unload_entry(hass: HomeAssistant, entry: OpenEPaperLinkConfigEntry) -> bool:
    """Unload the integration when removed or restarted.

    Handles both BLE and AP entries with appropriate cleanup.

    Args:
        hass: Home Assistant instance
        entry: Configuration entry being unloaded

    Returns:
        bool: True if unload was successful, False otherwise
    """

    if is_ble_entry(entry):
        # BLE device cleanup
        unload_ok = await hass.config_entries.async_unload_platforms(entry, BLE_PLATFORMS)
    else:
        # AP entry cleanup
        hub = entry.runtime_data
        unload_ok = await hass.config_entries.async_unload_platforms(entry, PLATFORMS)

        if unload_ok:
            await hub.shutdown()

    if unload_ok:
        entry.runtime_data = None

    return unload_ok


async def async_remove_entry(hass: HomeAssistant, entry: OpenEPaperLinkConfigEntry) -> None:
    """Handle complete removal of integration.

    Called when the integration is completely removed from Home Assistant
    (not during restarts). Performs cleanup of persistent storage files.

    Args:
        hass: Home Assistant instance
        entry: Configuration entry being removed
    """
    # Clear any repair issues for this entry
    if is_ble_entry(entry):
        # Clear BLE repair issues
        mac_address = entry.runtime_data["mac_address"]
        ir.async_delete_issue(hass, DOMAIN, f"ble_not_reachable_{mac_address}")
        ir.async_delete_issue(hass, DOMAIN, f"ble_protocol_error_{mac_address}")
        ir.async_delete_issue(hass, DOMAIN, f"ble_setup_error_{mac_address}")
    else:
        # Clear AP repair issue
        ir.async_delete_issue(hass, DOMAIN, "ap_unreachable")

    # Only remove shared storage files if this is the last config entry
    remaining_entries = [
        config_entry for config_entry in hass.config_entries.async_entries(DOMAIN)
        if config_entry.entry_id != entry.entry_id
    ]

    if not remaining_entries:
        # This was the last entry, safe to remove shared storage
        await async_remove_storage_files(hass)


async def async_remove_storage_files(hass: HomeAssistant) -> None:
    """Remove persistent storage files when removing integration.

    Cleans up files created by the integration:

    1. Tag types file (open_epaper_link_tagtypes.json)
    2. Tag storage file (.storage/open_epaper_link_tags)
    3. Image directory (www/open_epaper_link)

    This prevents orphaned files when the integration is removed
    and ensures a clean reinstallation if needed.

    Args:
        hass: Home Assistant instance
    """
    from .tag_types import reset_tag_types_manager

    # Remove tag types file
    tag_types_file = hass.config.path("open_epaper_link_tagtypes.json")
    if await hass.async_add_executor_job(os.path.exists, tag_types_file):
        try:
            await hass.async_add_executor_job(os.remove, tag_types_file)
            _LOGGER.debug("Removed tag types file")
        except OSError as err:
            _LOGGER.error("Error removing tag types file: %s", err)

    # Remove tag storage file
    storage_dir = hass.config.path(".storage")
    tags_file = os.path.join(storage_dir, f"{DOMAIN}_tags")
    if await hass.async_add_executor_job(os.path.exists, tags_file):
        try:
            await hass.async_add_executor_job(os.remove, tags_file)
            _LOGGER.debug("Removed tag storage file")
        except OSError as err:
            _LOGGER.error("Error removing tag storage file: %s", err)

    # Remove image directory
    image_dir = hass.config.path("www/open_epaper_link")
    if await hass.async_add_executor_job(os.path.exists, image_dir):
        try:
            # Get file list in executor
            files = await hass.async_add_executor_job(os.listdir, image_dir)

            # Remove each file in executor
            for file in files:
                file_path = os.path.join(image_dir, file)
                await hass.async_add_executor_job(os.remove, file_path)

            # Remove directory in executor
            await hass.async_add_executor_job(os.rmdir, image_dir)
            _LOGGER.debug("Removed image directory")
        except OSError as err:
            _LOGGER.error("Error removing image directory: %s", err)

    # Reset the tag types manager singleton since its storage was deleted
    reset_tag_types_manager()
    _LOGGER.debug("Reset tag types manager singleton")
