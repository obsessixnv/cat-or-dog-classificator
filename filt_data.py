import os
import shutil
from sklearn.model_selection import train_test_split

# Define paths
cat_dir = 'Cat'
dog_dir = 'Dog'
base_dir = 'data'

train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')

# Create directories
os.makedirs(train_dir, exist_ok=True)
os.makedirs(validation_dir, exist_ok=True)

train_cats_dir = os.path.join(train_dir, 'cats')
train_dogs_dir = os.path.join(train_dir, 'dogs')
os.makedirs(train_cats_dir, exist_ok=True)
os.makedirs(train_dogs_dir, exist_ok=True)

validation_cats_dir = os.path.join(validation_dir, 'cats')
validation_dogs_dir = os.path.join(validation_dir, 'dogs')
os.makedirs(validation_cats_dir, exist_ok=True)
os.makedirs(validation_dogs_dir, exist_ok=True)

# List all images
all_cats = [f for f in os.listdir(cat_dir) if os.path.isfile(os.path.join(cat_dir, f))]
all_dogs = [f for f in os.listdir(dog_dir) if os.path.isfile(os.path.join(dog_dir, f))]

# Split into train and validation sets (80% train, 20% validation)
train_cats, val_cats = train_test_split(all_cats, test_size=0.2, random_state=42)
train_dogs, val_dogs = train_test_split(all_dogs, test_size=0.2, random_state=42)

# Move files to appropriate directories
for fname in train_cats:
    src = os.path.join(cat_dir, fname)
    dst = os.path.join(train_cats_dir, fname)
    shutil.copyfile(src, dst)

for fname in val_cats:
    src = os.path.join(cat_dir, fname)
    dst = os.path.join(validation_cats_dir, fname)
    shutil.copyfile(src, dst)

for fname in train_dogs:
    src = os.path.join(dog_dir, fname)
    dst = os.path.join(train_dogs_dir, fname)
    shutil.copyfile(src, dst)

for fname in val_dogs:
    src = os.path.join(dog_dir, fname)
    dst = os.path.join(validation_dogs_dir, fname)
    shutil.copyfile(src, dst)

print("Files have been successfully organized into training and validation directories.")
