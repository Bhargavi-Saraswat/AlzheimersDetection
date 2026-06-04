# src/model.py
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models, optimizers

def build_cnn(input_shape=(128, 128, 3), num_classes=4):
    """
    Transfer learning with MobileNetV2 pretrained on ImageNet.
    Far superior to a 3-layer custom CNN for medical image classification.
    """
    base_model = MobileNetV2(
        input_shape=input_shape,
        include_top=False,       # remove ImageNet classification head
        weights='imagenet'       # use pretrained weights
    )
    base_model.trainable = False # freeze base in phase 1

    inputs = layers.Input(shape=input_shape)
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs, outputs)
    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-3),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model, base_model  # return both for fine-tuning