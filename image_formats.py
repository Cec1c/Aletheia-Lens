"""Image format declarations shared by the GUI and processing entry points."""


SUPPORTED_IMAGE_EXTENSIONS = (
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tiff",
    ".webp",
)

IMAGE_FILE_DIALOG_PATTERN = ";".join(
    f"*{extension}" for extension in SUPPORTED_IMAGE_EXTENSIONS
)


def is_supported_image(filename):
    """Return whether *filename* has an image extension accepted by the app."""
    return str(filename).casefold().endswith(SUPPORTED_IMAGE_EXTENSIONS)
