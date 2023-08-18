def model_callback():

    model= EfficientNetB0(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))  # imagenet

    for layer in model.layers:
        layer.trainable = False

    for layer in model.layers[-3:-1]:
        layer.trainable = True

    # add new classifier layers
    flat1 = Flatten()(model.layers[-1].output)
    output = Dense(1, activation='sigmoid', kernel_regularizer=tf.keras.regularizers.l2())(flat1)
    # define new model
    model = Model(inputs=model.inputs, outputs=output)
    optimizer = keras.optimizers.Adam(lr=LEARNING_RATE)
    loss = keras.losses.binary_crossentropy
    model.compile(loss=loss,
                  optimizer=optimizer,
                  metrics=['accuracy'])

    return model
