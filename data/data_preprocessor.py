import os
from PIL import Image
import pandas as pd
from collections import defaultdict
from typing import List, Tuple, Union, Dict

VALID_EXTENSIONS = {'.png'}
VALID_CLASSES = {
    'nm', 'l-r0.5', 'l-l0.5', 'fb', 'a-r0.5', 'a-l0.5', 'a-r0', 'a-l0'
}


def is_valid_image(filename: str) -> bool:
    return any(filename.lower().endswith(ext) for ext in VALID_EXTENSIONS)


def parse_filename(filename: str, root: str) -> Union[Tuple[str, str, str, str, str, int], None]:
    """
    Parse filename like: 101_90_fb_02_Seq_3.png
    Returns: (subject_id, angle, gait_class, trial, root, seq_index)
    """
    if not is_valid_image(filename):
        return None

    name, _ = os.path.splitext(filename)
    parts = name.split('_')
    if len(parts) < 6:
        print(f"Ignoring {filename}, {len(parts)} parts")
        return None

    subject_id, angle, gait_class, trial, _, seq = parts

    if gait_class not in VALID_CLASSES:
        print(f"Ignoring {filename}, class {gait_class}")
        return None

    try:
        seq_index = int(seq)
    except ValueError:
        print(f"Ignoring {filename}, invalid sequence number {seq}")
        return None

    return subject_id, angle, gait_class, trial, root, seq_index


def collect_sequences(dataset_root: str) -> Dict[Tuple[str, str, str, str, str], List[Tuple[int, str]]]:
    """
    Walk the dataset and group image paths by (subject, angle, class, trial, subfolder).
    Returns a dictionary: {group_key: [(seq_index, image_path), ...]}
    """
    grouped = defaultdict(list)

    for root, _, files in os.walk(dataset_root):
        for fname in files:
            parsed = parse_filename(fname, root)
            if parsed is None:
                continue
            subject_id, angle, gait_class, trial, root_path, seq_index = parsed
            group_key = (subject_id, angle, gait_class, trial, root_path)
            full_path = os.path.join(root, fname)
            grouped[group_key].append((seq_index, full_path))

    return grouped


def load_sequence(frames: List[Tuple[int, str]], load_images: bool = False) -> List[Union[str, Image.Image]]:
    """
    Sort frames by sequence index and return a list of image paths or PIL images.
    """
    frames_sorted = sorted(frames, key=lambda x: x[0])
    if load_images:
        return [Image.open(path).convert("L") for _, path in frames_sorted]
    return [path for _, path in frames_sorted]


def build_dataframe(grouped_sequences: Dict, load_images: bool = False, include_metadata: bool = True, min_sequence_length: int = None) -> pd.DataFrame:
    """
    Create DataFrame with columns:
        - sequence: list of frames (paths or images)
        - label: gait class (e.g., 'nm', 'fb')
        - (optional) subject, angle, trial, source (subfolder)
    Args:
        grouped_sequences (Dict): Mapping of metadata keys to frame lists
        load_images (bool): If True, load actual images. Else, use paths.
        include_metadata (bool): Whether to include subject, angle, etc.
        min_sequence_length (int): If provided, filter out sequences shorter than this
    """
    data = []
    for key, frames in grouped_sequences.items():
        subject_id, angle, gait_class, trial, subfolder = key
        sequence = load_sequence(frames, load_images)

        if min_sequence_length is not None and len(sequence) < min_sequence_length:
            continue  # Skip sequences that are too short

        entry = {
            'sequence': sequence,
            'label': gait_class
        }

        if include_metadata:
            entry.update({
                'subject': subject_id,
                'angle': angle,
                'trial': trial,
                'source': os.path.basename(subfolder)
            })

        data.append(entry)

    return pd.DataFrame(data)
def load_gait_frames(
    dataset_root: str,
    load_images: bool = False,
    include_metadata: bool = True,
) -> pd.DataFrame:
    """
    Load each frame as a separate row with its label (and optional metadata).
    Returns a DataFrame with columns: image (path or PIL), label, [subject, angle, trial, source, seq_index]
    """
    data = []
    for root, _, files in os.walk(dataset_root):
        for fname in files:
            parsed = parse_filename(fname, root)
            if parsed is None:
                continue
            subject_id, angle, gait_class, trial, subfolder, seq_index = parsed
            img_path = os.path.join(root, fname)
            img = Image.open(img_path).convert("L") if load_images else img_path

            entry = {
                'image': img,
                'label': gait_class
            }
            if include_metadata:
                entry.update({
                    'subject': subject_id,
                    'angle': angle,
                    'trial': trial,
                    'source': os.path.basename(subfolder),
                    'seq_index': seq_index
                })
            data.append(entry)
    return pd.DataFrame(data)

def load_gait_sequences(
    dataset_root: str,
    load_images: bool = False,
    include_metadata: bool = True,
    min_sequence_length: int = None,
) -> pd.DataFrame:
    """
    Main function to load and structure gait dataset into a DataFrame.
    Each row corresponds to a single sequence.
    """
    grouped = collect_sequences(dataset_root)
    return build_dataframe(grouped, load_images, include_metadata, min_sequence_length=min_sequence_length)

if __name__ == '__main__':
    df = load_gait_frames(r"D:\GEI\processed_dataset\stage1_binary", load_images=False)
    print(df.columns)
    print(df.sample(10))
    print("Total frames:", len(df))
    print("Class distribution:\n", df['label'].value_counts())