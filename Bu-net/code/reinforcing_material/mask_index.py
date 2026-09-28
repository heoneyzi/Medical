"""Find complete sets of user-exported segmentation images."""
from pathlib import Path
import re

SERIES = (
    ('ground_truth', 'Ground truth'),
    ('model_prediction', 'Model 1'),
    ('model2_prediction', 'Model 2'),
    ('model3_prediction', 'Model 3'),
)


def discover_cases(directory):
    """Return complete positive case IDs and the IDs found for each series."""
    directory = Path(directory)
    available = {prefix: set() for prefix, _ in SERIES}
    if directory.is_dir():
        pattern = re.compile(
            r'^(ground_truth|model_prediction|model2_prediction|model3_prediction)_([1-9][0-9]*)\.png$'
        )
        for path in directory.iterdir():
            match = pattern.fullmatch(path.name)
            if match and path.is_file():
                available[match.group(1)].add(int(match.group(2)))
    complete = sorted(set.intersection(*available.values()))
    return complete, available


def load_case_images(directory, case_id):
    """Load images eagerly so corrupt/missing files can be handled by the UI."""
    from PIL import Image

    images = []
    for prefix, label in SERIES:
        with Image.open(Path(directory) / f'{prefix}_{case_id}.png') as image:
            images.append((label, image.copy()))
    return images
