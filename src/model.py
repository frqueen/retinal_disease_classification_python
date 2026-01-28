from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import Adam

def make_model():
    base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=(224, 224, 3), pooling='avg')
    
    # Fine-tune deeper layers only
    for layer in base_model.layers[:83]:
        layer.trainable = False
    for layer in base_model.layers[83:]:
        layer.trainable = True

    inputs = Input(shape=(224, 224, 3))
    x = base_model(inputs, training=False)
    x = BatchNormalization()(x)
    x = Dense(1024, activation='relu', kernel_regularizer=regularizers.l2(0.001))(x)
    x = Dropout(0.45)(x)
    x = Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.001))(x)
    x = Dropout(0.3)(x)
    outputs = Dense(4, activation='softmax')(x)
        model = Model(inputs, outputs)
    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

model = make_model()