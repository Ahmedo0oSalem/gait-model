import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
import pandas as pd

from data.data_preprocessor import load_gait_sequences
from utils.visualize_samples import visualize_optical_flow_samples, visualize_gait_sequence_samples

class GaitFrameSequenceDataset(Dataset):
    def __init__(self, dataframe, transform=None, label_to_index=None,
                 return_metadata=False, train_augmentations=None, image_size=(224, 224)):
        self.data = dataframe
        self.transform = transform or transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        self.train_augmentations = train_augmentations
        self.return_metadata = return_metadata
        self.image_size = image_size

        self.target_length = int(np.median(self.data['sequence'].apply(len)))

        if label_to_index is None:
            classes = sorted(self.data['label'].unique())
            self.label_to_index = {cls: i for i, cls in enumerate(classes)}
        else:
            self.label_to_index = label_to_index

    def __len__(self):
        return len(self.data)

    def load_and_transform_image(self, path):
        img = Image.open(path).convert("L")
        if self.train_augmentations is not None:
            return self.train_augmentations(img)
        return self.transform(img)

    def pad_or_downsample(self, frames):
        if len(frames) < self.target_length:
            pad = [torch.zeros_like(frames[0])] * (self.target_length - len(frames))
            return pad + frames
        elif len(frames) > self.target_length:
            indices = np.linspace(0, len(frames) - 1, self.target_length).astype(int)
            return [frames[i] for i in indices]
        return frames

    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        label = self.label_to_index[item['label']]
        paths = item['sequence']
        frames = [self.load_and_transform_image(p) for p in paths]
        frames = self.pad_or_downsample(frames)
        frames = torch.stack(frames)  # [T, 1, H, W]

        if self.return_metadata:
            meta = {k: item[k] for k in ['subject', 'angle', 'trial', 'source']}
            return frames, label, len(paths), meta
        return frames, label, len(paths)

class GaitFrameDataset(Dataset):
    def __init__(self, dataframe, transform=None, label_to_index=None,
                 return_metadata=False, train_augmentations=None, image_size=(224, 224),aggregate_lr_labels=False):
        # Step 1: Flatten the dataframe so each frame is its own row
        rows = []
        for _, row in dataframe.iterrows():
            for path in row['sequence']:
                entry = {
                    'image_path': path,
                    'label': row['label']
                }
                if 'subject' in row:
                    entry['subject'] = row['subject']
                if 'angle' in row:
                    entry['angle'] = row['angle']
                if 'trial' in row:
                    entry['trial'] = row['trial']
                if 'source' in row:
                    entry['source'] = row['source']
                rows.append(entry)
        self.data = pd.DataFrame(rows)

        # Step 2: Define transforms
        self.transform = transform or transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        self.train_augmentations = train_augmentations
        self.return_metadata = return_metadata

        if aggregate_lr_labels:
            aggregation_map = {
                'a-l0': 'arm', 'a-r0': 'arm', 'a-l0.5': 'arm', 'a-r0.5': 'arm',
                'l-l0.5': 'leg', 'l-r0.5': 'leg',
                'nm': 'nm',
                'fb': 'fb'
            }
            # Replace original labels in dataframe with aggregated classes
            self.data['label'] = self.data['label'].apply(lambda x: aggregation_map.get(x, x))

    # Then proceed with label encoding as usual:
        # Step 3: Label encoding
        if label_to_index is None:
            classes = sorted(self.data['label'].unique())
            self.label_to_index = {cls: i for i, cls in enumerate(classes)}
        else:
            self.label_to_index = label_to_index


    def __len__(self):
        return len(self.data)

    def load_and_transform_image(self, path):
        img = Image.open(path).convert("L")
        if self.train_augmentations is not None:
            img = self.train_augmentations(img)
        else:
            img = self.transform(img)
        return img

    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        img = self.load_and_transform_image(item['image_path'])
        label = self.label_to_index[item['label']]

        if self.return_metadata:
            meta = {k: item[k] for k in ['subject', 'angle', 'trial', 'source'] if k in item}
            return img, label, meta
        return img, label
    




from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
import torch
import cv2


class GaitOpticalFlowDataset(Dataset):
    def __init__(self, dataframe, transform=None, label_to_index=None,
                 return_metadata=False, image_size=(224, 224), train_augmentations=None, use_tvl1=False):
        self.data = dataframe
        self.transform = transform or transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        self.image_size = image_size
        self.return_metadata = return_metadata
        self.flow_augment = train_augmentations
        self.use_tvl1 = use_tvl1
        self.target_flow_length = int(np.median(self.data['sequence'].apply(len))) - 1

        if label_to_index is None:
            classes = sorted(self.data['label'].unique())
            self.label_to_index = {cls: i for i, cls in enumerate(classes)}
        else:
            self.label_to_index = label_to_index



    def __len__(self):
        return len(self.data)

    def preprocess_image(self, pil_img):
        img_np = np.array(pil_img)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(img_np)
        return Image.fromarray(enhanced, mode="L")

    def load_and_transform_image(self, path):
        img = Image.open(path).convert("L")
        img = self.preprocess_image(img)
        return self.transform(img)

    def pad_or_downsample_flow(self, flows):
        if len(flows) < self.target_flow_length:
            pad = [torch.zeros_like(flows[0])] * (self.target_flow_length - len(flows))
            return pad + flows
        elif len(flows) > self.target_flow_length:
            idx = np.linspace(0, len(flows) - 1, self.target_flow_length).astype(int)
            return [flows[i] for i in idx]
        return flows

    def compute_optical_flow(self, f1, f2):
        f1 = (f1.squeeze(0) * 255).numpy().astype(np.uint8)
        f2 = (f2.squeeze(0) * 255).numpy().astype(np.uint8)

        if self.use_tvl1:
            tv_l1 = cv2.optflow.DualTVL1OpticalFlow_create()
            flow = tv_l1.calc(f1, f2, None)
            flow = np.clip(flow, -20, 20) / 20.0
        else:
            flow = cv2.calcOpticalFlowFarneback(
                f1, f2, None,
                pyr_scale=0.5, levels=5,
                winsize=11, iterations=5,
                poly_n=5, poly_sigma=1.1,
                flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN
            )
            flow = flow / (np.max(np.abs(flow)) + 1e-6)  # Normalize to [-1, 1]

        return torch.from_numpy(flow.transpose(2, 0, 1)).float()

    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        label = self.label_to_index[item['label']]
        paths = item['sequence']

        frames = [self.load_and_transform_image(p) for p in paths]
        original_seq_len = len(frames)

        flow_sequence = []
        for i in range(len(frames) - 1):
            flow = self.compute_optical_flow(frames[i], frames[i + 1])
            if self.flow_augment:
                flow = self.flow_augment(flow)
            flow_sequence.append(flow)

        flow_sequence = self.pad_or_downsample_flow(flow_sequence)
        flow_tensor = torch.stack(flow_sequence)

        if self.return_metadata:
            meta = {k: item[k] for k in ['subject', 'angle', 'trial', 'source']}
            return flow_tensor, label, original_seq_len, meta

        return flow_tensor, label, original_seq_len


# if __name__ == "__main__":
#     flow_or_sequence = 1 # 1 to test sequence, 0 to test flow
#     if flow_or_sequence == 1:
#         df = load_gait_sequences("E:\GEI\gait-model\data\Multiclass6", load_images=False)

#         dataset = GaitFrameSequenceDataset(
#             dataframe=df,
#             return_metadata=False
#         )

#         loader = DataLoader(dataset, batch_size=2)
#         seq, label, lengths = next(iter(loader))
#         print("Temporal sequence shape:", seq.shape)  # Expected: [B, T, 1, H, W]

#         # Optional visualization
#         visualize_gait_sequence_samples(dataset, n=5)
#     else:
#         df = load_gait_sequences("./gei_maps/Multiclass6", load_images=False)

#         dataset = GaitOpticalFlowDataset(
#             dataframe=df,
#             return_metadata=False,
#             use_tvl1=True
#         )

#         loader = DataLoader(dataset, batch_size=2)
#         flow, label, lengths = next(iter(loader))
#         print("Optical flow shape:", flow.shape)  # Expected: [B, 2, H, W]

#         # Optional visualization
#         visualize_optical_flow_samples(dataset, n=5)


#for the eidted gait sequance 
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from data.data_preprocessor import load_gait_sequences

    # Load the sequences (each row is a sequence)
    df = load_gait_sequences(r"D:\GEI\gait-model\data\Multiclass6", load_images=False)

    # Create the frame-by-frame dataset
    dataset = GaitFrameDataset(
        dataframe=df,
        return_metadata=True
    )

    # Show what the flattened dataframe looks like
    print("\n=== Flattened DataFrame Preview ===")
    print(dataset.data.head(10))  # show first 10 frame entries
    print(f"\nTotal frames: {len(dataset)}")

    # Create a DataLoader
    loader = DataLoader(dataset, batch_size=8, shuffle=True)

    # Get one batch
    imgs, labels, metas = next(iter(loader))
    print("\nBatch of frames:", imgs.shape)  # [8, 1, 224, 224]
    print("Labels:", labels)
    print("Metadata sample:", metas)

    # --- Visualization ---
    # Convert 8 frames to numpy and display them
    plt.figure(figsize=(12, 4))
    for i in range(min(8, len(imgs))):
        img = imgs[i].squeeze(0).numpy()
        label_idx = labels[i].item() if torch.is_tensor(labels[i]) else labels[i]
        label_name = [k for k, v in dataset.label_to_index.items() if v == label_idx][0]
        
        # Access meta info for this frame
        subject = metas['subject'][i]
        angle = metas['angle'][i]

        plt.subplot(2, 4, i + 1)
        plt.imshow(img, cmap='gray')
        plt.axis('off')
        plt.title(f"{label_name}\nSubj {subject}, Angle {angle}")

    plt.tight_layout()
    plt.show()
