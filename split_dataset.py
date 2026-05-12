import os
import shutil
import random
from sklearn.model_selection import train_test_split

# Path to original dataset
source_dir = "dataset"

# Path for split dataset
output_dir = "split_dataset"

classes = ["healthy", "gill", "red_spot", "eus"]

# Create output folders
for split in ["train", "validation", "test"]:
    for cls in classes:
        os.makedirs(os.path.join(output_dir, split, cls), exist_ok=True)

# Split images
for cls in classes:

    cls_path = os.path.join(source_dir, cls)

    images = os.listdir(cls_path)

    random.shuffle(images)

    # 70% train
    train_imgs, temp_imgs = train_test_split(
        images,
        test_size=0.30,
        random_state=42
    )

    # 15% validation, 15% test
    val_imgs, test_imgs = train_test_split(
        temp_imgs,
        test_size=0.50,
        random_state=42
    )

    # Function to copy files
    def copy_files(img_list, split_name):

        for img in img_list:

            src = os.path.join(cls_path, img)

            dst = os.path.join(
                output_dir,
                split_name,
                cls,
                img
            )

            shutil.copy(src, dst)

    copy_files(train_imgs, "train")
    copy_files(val_imgs, "validation")
    copy_files(test_imgs, "test")

print("Dataset split completed successfully!")