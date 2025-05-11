def train_new(use_val_set=True, epochs_add=10, batch_size_previous=25):
    cleanup_gpu_memory()
    
    try:
        # train_df, val_df, le = load_and_preprocess_data(random_state=42)
        # num_classes = len(le.classes_)
        # print("Classes: ", num_classes)

        # Load model 
        with open(config['label_encoder_path'], 'rb') as f:
            classes = np.load(f, allow_pickle=True)
        le = LabelEncoder()
        le.classes_ = classes
        num_classes = len(le.classes_)
        print(le)

        train_df = pd.read_csv(config["train_set_csv"]) if use_val_set else load_and_preprocess_data(save_splits=False)[0]
        print(f"Evaluating on {len(train_df)} samples")

        val_df = pd.read_csv(config["val_set_csv"]) if use_val_set else load_and_preprocess_data(save_splits=False)[1]
        print(f"Evaluating on {len(val_df)} samples")
        
        # Create the mopdel
        input_shape = config["input_shape"] 
        model = tf.keras.models.load_model(config["best_model"])

        train_gen = RiceDataGenerator(
            df=train_df,
            base_path=config["data_path"],
            batch_size=25,
            target_size=config["target_size"],
            shuffle=False,
            debug=True
        )
        
        val_gen = RiceDataGenerator(
            df=val_df,
            base_path=config["data_path"],
            batch_size=25,
            target_size=config["target_size"],
            shuffle=False,
            debug=False
        )        

        run = 0  # Initialize run counter somewhere outside this training loop

        # Inside your training code:
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=5,
                mode='max',
                restore_best_weights=True  
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_accuracy',
                factor=0.5,
                patience=3,
                mode='max'
            )
        ]

        # Only add ModelCheckpoint every 5 runs
        if run % 3 == 0:
            callbacks.append(
                tf.keras.callbacks.ModelCheckpoint(
                    config["best_model"],
                    save_weights_only=False,
                    monitor='val_accuracy',
                    save_best_only=True
                )
            )

        continued_history = best_model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=epochs_add,
            initial_epoch=history.epoch[-1] + 1 if 'history' in locals() else 0,
            callbacks=callbacks
        )

        run += 1  # Increment run counter after each training
        
        plt.plot(continued_history.history['accuracy'], label='Train Accuracy')
        plt.plot(continued_history.history['val_accuracy'], label='Val Accuracy')
        plt.axhline(y=max(continued_history.history['val_accuracy']), color='r', linestyle='--', label='Best Val Accuracy')
        plt.legend()
        
        return model, continued_history
        
    except Exception as e:
        print(f"Training failed: {e}")
        cleanup_gpu_memory()
        raise

continue_model, continue_model_history = train_new(use_val_set=True, epochs_add=4, batch_size_previous=25)