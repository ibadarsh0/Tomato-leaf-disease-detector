import tensorflow as tf

IMG_SIZE = (128, 128)
BATCH_SIZE = 32

def load_datasets():
    train_ds = tf.keras.utils.image_dataset_from_directory(
        "Dataset/Train", image_size=IMG_SIZE, batch_size=BATCH_SIZE, label_mode='categorical'
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        "Dataset/Validation", image_size=IMG_SIZE, batch_size=BATCH_SIZE, label_mode='categorical'
    )
    test_ds = tf.keras.utils.image_dataset_from_directory(
        "Dataset/Test", image_size=IMG_SIZE, batch_size=BATCH_SIZE, label_mode='categorical'
    )
    return train_ds, val_ds, test_ds
