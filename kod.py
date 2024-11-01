
from sklearn.metrics import precision_score, recall_score, f1_score

# Farklı optimizers ve batch_size için test fonksiyonu
def test_with_different_params(model_type='binary', optimizer='adam', batch_size=32):
    # Model seçimi
    if model_type == 'binary':
        model = create_amazon_review_model()  # Binary sınıflandırma modeli
        y_train_labels = y_train
        y_test_labels = y_test
        loss = 'binary_crossentropy'
    else:
        model = create_multiclass_amazon_review_model()  # Multiclass sınıflandırma modeli
        y_train_labels = y_train_multiclass
        y_test_labels = y_test_multiclass
        loss = 'categorical_crossentropy'
    
    # Model derleme (farklı optimizer ve loss ile)
    model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
    
    # Modeli eğit
    history = model.fit(X_train_pad, y_train_labels, epochs=5, batch_size=batch_size, validation_split=0.2, verbose=1)
    
    # Tahminler ve değerlendirme
    if model_type == 'binary':
        y_pred = (model.predict(X_test_pad) > 0.5).astype('int32')
        y_true = y_test
    else:
        y_pred = np.argmax(model.predict(X_test_pad), axis=1)
        y_true = np.argmax(y_test_multiclass.values, axis=1)

    # Performans metrikleri
    precision = precision_score(y_true, y_pred, average='binary' if model_type == 'binary' else 'macro')
    recall = recall_score(y_true, y_pred, average='binary' if model_type == 'binary' else 'macro')
    f1 = f1_score(y_true, y_pred, average='binary' if model_type == 'binary' else 'macro')

    print(f"{model_type.capitalize()} - Optimizer: {optimizer}, Batch Size: {batch_size}")
    print(f"Precision: {precision:.2f}, Recall: {recall:.2f}, F1 Score: {f1:.2f}\n")

# Farklı batch size ve optimizer'larla test
test_with_different_params(model_type='binary', optimizer='adam', batch_size=32)
test_with_different_params(model_type='binary', optimizer='sgd', batch_size=64)
test_with_different_params(model_type='multiclass', optimizer='adam', batch_size=32)
test_with_different_params(model_type='multiclass', optimizer='sgd', batch_size=64)
