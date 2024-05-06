import tensorflow as tf
from keras.callbacks import EarlyStopping, ModelCheckpoint

# Define optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)


# Loss function (categorical cross-entropy)
def loss_function(real, pred):
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred) * mask
    return tf.reduce_mean(loss_)


# Training loop
def train_model(model, features, labels, validation_features, validation_labels, epochs=10):
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='min'),
        ModelCheckpoint('best_model.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    ]
    model.compile(loss=loss_function, optimizer=optimizer, metrics=['accuracy'])
    model.fit(x=features, y=labels, epochs=epochs,
              validation_data=(validation_features, validation_labels), callbacks=callbacks)
