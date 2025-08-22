from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, accuracy_score
import numpy as np

model = load_model("hand_sign_model.h5")

test_dir = "D:\expo\handsign\Test"

test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(64, 64),
    batch_size=32,
    class_mode="categorical",
    shuffle=False
)

y_pred = model.predict(test_generator)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = test_generator.classes

accuracy = accuracy_score(y_true, y_pred_classes)
print("Test Accuracy:", accuracy)

print(classification_report(y_true, y_pred_classes, target_names=list(test_generator.class_indices.keys())))