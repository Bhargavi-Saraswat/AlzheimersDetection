import os
import shutil
import random

# CONFIG
SOURCE_DIR = 'Data'  
DEST_DIR = 'data_split'
CLASSES = ['MildDemented', 'ModerateDemented', 'NonDemented', 'VeryMildDemented']
SPLIT_RATIOS = {'train': 0.7, 'val': 0.15, 'test': 0.15}

# Create output folders
for split in ['train', 'val', 'test']:
    for cls in CLASSES:
        path = os.path.join(DEST_DIR, split, cls)
        os.makedirs(path, exist_ok=True)

# Split data
for cls in CLASSES:
    src_folder = os.path.join(SOURCE_DIR, cls)
    print(f"🔍 Looking in: {src_folder}")  # Debug print

    if not os.path.exists(src_folder):
        print(f"❌ Folder not found: {src_folder}")
        continue  # Skip this class if folder is missing

    all_images = [img for img in os.listdir(src_folder) if img.lower().endswith(('.jpg', '.png', '.jpeg'))]
    random.shuffle(all_images)
    
    print(f"🖼️ Found {len(all_images)} images in {cls}")

    total = len(all_images)
    train_count = int(total * SPLIT_RATIOS['train'])
    val_count = int(total * SPLIT_RATIOS['val'])

    split_data = {
        'train': all_images[:train_count],
        'val': all_images[train_count:train_count + val_count],
        'test': all_images[train_count + val_count:]
    }

    for split in ['train', 'val', 'test']:
        for img in split_data[split]:
            src_path = os.path.join(src_folder, img)
            dst_path = os.path.join(DEST_DIR, split, cls, img)
            shutil.copy2(src_path, dst_path)

print("✅ Dataset split complete.")