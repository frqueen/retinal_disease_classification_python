from keras import callbacks

early_stop = callbacks.EarlyStopping(
    monitor="val_accuracy", patience=20, restore_best_weights=True, verbose=1
)

reduce_lr = callbacks.ReduceLROnPlateau(
    monitor='val_accuracy', factor=0.5, patience=5, min_lr=1e-6, verbose=1
)

checkpoint = callbacks.ModelCheckpoint(
    'best_model.keras', monitor='val_accuracy', save_best_only=True, verbose=1
)
history = model.fit(
    train,
    validation_data=val,
    epochs=50,
    callbacks=[early_stop, reduce_lr, checkpoint]
)