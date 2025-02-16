%pip install imutils
# General purpose libraries
import pandas as pd
import numpy as np
import os
import sys
import imutils
import itertools
import shutil
import random
from collections import  Counter

# Visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns

# Sklearn libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

# OpenCv libraries
import cv2
from PIL import Image

# Tensorflow libraries
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Dropout, Flatten, GlobalAveragePooling2D
from tensorflow.keras.applications import ResNet50,ResNet152
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import load_model


# Image labels
img_labels = ['glioma','meningioma','notumor','pituitary']

# Train & directory
train_dir = "/archive/Training"
test_dir  = "/archive/Testing"

# Initialize inputs and target/labels list
X_Images = []
Y_Labels = []

# To find the images height, weidth and their pixel values information
pixel_min = float('inf')
pixel_max = float('-inf')

height_min = float('inf')
height_max = float('-inf')

weidth_min = float('inf')
weidth_max = float('-inf')

# Read all images from both the directories and
# store them with their respective labels
for dir in [train_dir, test_dir]:
    # dir is the current directory
    for lab in img_labels:
        # for each labels the folder path is:
        folder_path = os.path.join(dir, lab)
        # for current folder in current directory, images are:
        for img in os.listdir(folder_path):
            # image path:
            img_path = os.path.join(folder_path, img)
            # Read current image using opencv cv2
            image = cv2.imread(img_path)
            # if we did not read invalid image file
            if image is not None:
                # updates pixels min and max
                pixel_max = max(pixel_max, image.max())
                pixel_min = min(pixel_min, image.min())

                # Updates image heights and weidths
                height_max = max(height_max, image.shape[0])
                height_min = min(height_min, image.shape[0])

                weidth_max = max(weidth_max, image.shape[1])
                weidth_min = min(weidth_min, image.shape[1])

                # Update the image and labels list
                X_Images.append(image)
                Y_Labels.append(lab)
            else:
                print('Invalid Image!')


# Succeful message
print("Data reading done.....")


#print the pixels information
print(f'Maximum pixel value: {pixel_max}')
print(f'Minimum pixel value: {pixel_min}')

# Print the image height and weidth informations
print(f'Maximum height value: {height_max}')
print(f'Minimum height value: {height_min}')

print(f'Maximum weidth value: {weidth_max}')
print(f'Minimum weidth value: {weidth_min}')

# Shuffles the images and their respective labels

# for better generalization
np.random.seed(101)
# Shuffle both arrays in unison using a common random seed
indices = np.arange(len(X_Images))  # list of indices
np.random.shuffle(indices)   # shuffle the indices
# shuffling the images and its labels
X_images = [X_Images[idx] for idx in indices]
Y_labels = [Y_Labels[idx] for idx in indices]

# Succesful messages
print("Reshaping and numpy array conversion and shuffling has done!")


num_img = 15
# Create a 2x5 grid for subplots (2 rows, 5 columns)
plt.figure(figsize=(15, 6))
# Randomly select 15 numbers from 0 to 7022
random_numbers = random.sample(range(7023), 15)
for i in range(num_img):
    # Set the position of each subplot, using i + 1 for the index
    plt.subplot(num_img//5, 5, i+1)  # 3 rows, 5 columns, position i+1
    plt.imshow(X_images[random_numbers[i]], cmap='gray')  # Display the image
    plt.title(f'{Y_labels[random_numbers[i]]}')  # Set the title with the label
    plt.axis('off')  # Hide axis ticks and labels

plt.show()

# Function  to crop an image
def Crop_image(img):
    """
    Finds the extreme points on the image and crops the rectangular out of them
    """
    # Copy from original image
    img = img.copy()
    gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) # RGB to GRAY conversion
    # Smoothing images while preserving edges
    gray_img = cv2.GaussianBlur(gray_img, (3,3), 0)

    # Thresholding the image to remove unnecessary pixels,
    # then perfom a series of erosions +
    # dilations to remove any samll region of noise while preseving
    # important features in the images
    _,img_thresh = cv2.threshold(gray_img, 45, 255, cv2.THRESH_BINARY)
    img_erode = cv2.erode(img_thresh, None, iterations=2)
    img_dilate = cv2.dilate(img_erode, None, iterations=2)

    # Find contours in thresholded image, then grab the largest one
    cnts = cv2.findContours(img_dilate.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    c = max(cnts, key=cv2.contourArea)

    # Find the extreme points
    extLeft = tuple( c[c[:, :, 0].argmin()][0])
    extRight = tuple(c[c[:, :, 0].argmax()][0])
    extTop = tuple(c[c[:, :, 1].argmin()][0])
    extBot = tuple(c[c[:, :, 1].argmax()][0])

    # Add contour on the image
    img_cnt = cv2.drawContours(img.copy(), [c], -1, (0, 255, 255), 4)

    # Add extreme points
    img_pnt = cv2.circle(img_cnt.copy(), extLeft, 8, (0,0,255),-1)
    img_pnt = cv2.circle(img_pnt, extRight, 8, (0,255,0),-1)
    img_pnt = cv2.circle(img_pnt, extTop, 8, (255,0,0),-1)
    img_pnt = cv2.circle(img_pnt, extBot, 8, (255,0,0),-1)
    # Crop
    Add_pixel = 0
    img_crop = img[extTop[1]-Add_pixel:extBot[1]+Add_pixel, extLeft[0]-Add_pixel:extRight[0]+Add_pixel].copy()
    return img_cnt, img_pnt, img_crop # contourd image, image with extreme points, croped image


    # img = cv2.imread("/kaggle/input/brain-tumor-mri-dataset/Training/meningioma/Tr-meTr_0000.jpg")
img = X_Images[2]
img_cnt, img_pnt, img_crop = Crop_image(img)
img_crop = cv2.resize(img_crop, dsize=(224,224), interpolation=cv2.INTER_LANCZOS4)
plt.figure(figsize=(15,10))
plt.subplot(141)
plt.imshow(img)
plt.title('step1:Get the original image')
plt.axis('off')
plt.subplot(142)
plt.imshow(img_cnt)
plt.title('step2:Contoured image')
plt.axis('off')
plt.subplot(143)
plt.imshow(img_pnt)
plt.title('step3:Contoured image with extreme points')
plt.axis('off')
plt.subplot(144)
plt.imshow(img_crop)
plt.title('step4:Cropped image')
plt.axis('off')
plt.show()

img.shape,img_cnt.shape, img_crop.shape

X_crops = []
for img in X_images:
    # crop the current image
    _,_, img_crop = Crop_image(img)
    # resize it
    img_crop = cv2.resize(img_crop, dsize=(224,224), interpolation=cv2.INTER_LANCZOS4)
    # append into new list
    X_crops.append(img_crop)

print('Crop on original version has been done!')


num_img = 105
# Create a 2x5 grid for subplots (2 rows, 5 columns)
plt.figure(figsize=(50, 70))
# Randomly select 15 numbers from 0 to 7022
random_numbers = random.sample(range(7023), 105)
for i in range(num_img):
    # Set the position of each subplot, using i + 1 for the index
    plt.subplot(num_img//5, 5, i+1)  # 3 rows, 5 columns, position i+1
    plt.imshow(X_crops[random_numbers[i]], cmap='gray')  # Display the image
    plt.title(f'{Y_labels[random_numbers[i]]}')  # Set the title with the label
    plt.axis('off')  # Hide axis ticks and labels

plt.show()

X_dnoise = []
for img in X_crops:
    dnoise_img = cv2.bilateralFilter(img, 2, 50,50)
    # apply colormap bone for highlight important features
    cmap_img = cv2.applyColorMap(dnoise_img, cv2.COLORMAP_BONE)
    X_dnoise.append(cmap_img)
print('Denoising has done!')

num_img = 15
# Create a 2x5 grid for subplots (2 rows, 5 columns)
plt.figure(figsize=(15, 8))
# Randomly select 15 numbers from 0 to 7022
random_numbers = random.sample(range(7023), 15)
for i in range(num_img):
    # Set the position of each subplot, using i + 1 for the index
    plt.subplot(num_img//5, 5, i+1)  # 3 rows, 5 columns, position i+1
    plt.imshow(X_dnoise[random_numbers[i]], cmap='gray')  # Display the image
    plt.title(f'{Y_labels[random_numbers[i]]}')  # Set the title with the label
    plt.axis('off')  # Hide axis ticks and labels

plt.show()


X_nimages = []
for img in X_dnoise:
    norm_img = (img/255.0).astype('float32')
    X_nimages.append(norm_img)
# Successful msg
print('Normalized has done!')


# Convert all the list into np array
X_nimages = np.array(X_nimages)
Y_labels = np.array(Y_labels)


# Counts for each types of tumor:
pd.DataFrame(list(Counter(Y_labels).items()), columns=['Class', 'Count'])


#Plot using Seaborn's countplot with 'viridis' palette
plt.figure(figsize=(10, 6))
sns.countplot(y=Y_labels, palette='viridis')  # Use data frame with hue
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability if needed
plt.xlabel('Tumor Types')  # Label for the x-axis
plt.ylabel('Count')  # Label for the y-axis
plt.title('Distribution of Tumor Types')  # Title of the plot
plt.show()


#stratified splitting
X_train, X_test, y_train, y_test = train_test_split(
    X_nimages, Y_labels, test_size=0.2, stratify=Y_labels, random_state=42
)

print('Spliting into 80-20 has done!')


# Distribution of class in training set
display(pd.DataFrame(list(Counter( y_train).items()), columns=['Class', 'Count']))

# Barplots
#Plot using Seaborn's countplot with 'viridis' palette
plt.figure(figsize=(10, 6))
sns.countplot(y= y_train, palette='viridis')  # Use data frame with hue
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability if needed
plt.xlabel('Tumor Types')  # Label for the x-axis
plt.ylabel('Count')  # Label for the y-axis
plt.title('Distribution of Tumor Types')  # Title of the plot
plt.show()


# Get class distribution using Counter
class_distribution = Counter(y_train)

# Find the maximum samples among all classes for balancing
max_samples = max(class_distribution.values())

# Find the class with the maximum number of samples
major_class = max(class_distribution, key=class_distribution.get)

print(f"The class with the most samples is: {major_class} with {max_samples} samples.")

class_distribution

# Step 1: Calculate class counts
class_counts = {class_name[0]: class_name[1]  for class_name in class_distribution.items()}
print(class_counts)

# Define ImageDataGenerator for augmentation
datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.05,
    height_shift_range=0.05,
    horizontal_flip=True,
    fill_mode='nearest'
)
# Define class names
class_names = ['meningioma', 'glioma', 'notumor', 'pituitary']

# Step 1: Calculate class counts
class_counts = {class_name[0]: class_name[1]  for class_name in class_distribution.items()}

# Step 2: Loop through each class and augment if the class is a minority
augmented_images = []
augmented_labels = []

for class_name in class_names:
    # Step 2.1: Filter images for the current class
    class_images = [X_train[i] for i in range(len(y_train)) if y_train[i] == class_name]

    # Step 2.2: If the class is a minority class, perform augmentation
    if class_counts[class_name] < max_samples:
        num_images_to_select = max_samples - class_counts[class_name]  # How many images to augment
        # Randomly select images from the class to augment
        selected_images = random.sample(class_images, num_images_to_select)
        # Step 2.3: Augment selected images
        for img in selected_images:
            img = img_to_array(img) if not isinstance(img, np.ndarray) else img  # Convert to numpy array if not already
            img = img.reshape((1,) + img.shape)  # Reshape for grayscale (1, height, width, 3)

            # Generate one augmented image per original image
            for batch in datagen.flow(img, batch_size=1):
                augmented_image = batch[0].astype('float32')  # Convert back to float64 format
                augmented_images.append(augmented_image)  # Append the reshaped image
                augmented_labels.append(class_name)  # Append corresponding label
                break  # Only generate one augmented image per original image
        print(f"Augmented {num_images_to_select} images for the class {class_name}.")



num_img = 15
# Create a 2x5 grid for subplots (2 rows, 5 columns)
plt.figure(figsize=(15, 8))
# Randomly select 15 numbers from 0 to 7022
random_numbers = random.sample(range(len(augmented_images)), 15)
for i in range(num_img):
    # Set the position of each subplot, using i + 1 for the index
    plt.subplot(num_img//5, 5, i+1)  # 3 rows, 5 columns, position i+1
    plt.imshow(augmented_images[random_numbers[i]])  # Display the image
    plt.title(f'{augmented_labels[random_numbers[i]]}')  # Set the title with the label
    plt.axis('off')  # Hide axis ticks and labels

plt.show()

# Add augmented images and labels to the dataset
X_train_augmented = np.concatenate((X_train, np.array(augmented_images)))
y_train_augmented = np.concatenate((y_train, np.array(augmented_labels)))
print(f"New dataset size: {len(X_train_augmented)} images, {len(y_train_augmented)} labels.")

# Shuffle then for better generalization
indices = np.arange(len(X_train_augmented))  # list of indices
np.random.shuffle(indices)
X_train_augmented = X_train_augmented[indices]
y_train_augmented = y_train_augmented[indices]
print('Shuffle has done!')

# Distribution of class in training set
display(pd.DataFrame(list(Counter( y_train_augmented).items()), columns=['Class', 'Count']))

# Barplots
#Plot using Seaborn's countplot with 'viridis' palette
plt.figure(figsize=(10, 6))
sns.countplot(y= y_train_augmented, palette='viridis')  # Use data frame with hue
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability if needed
plt.xlabel('Tumor Types')  # Label for the x-axis
plt.ylabel('Count')  # Label for the y-axis
plt.title('Distribution of Tumor Types after augmentation')  # Title of the plot
plt.show()

# Numerical conversion  usingsklearn LabelEncoder
# Label Encoding (convert string labels to integers)
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train_augmented) # fit on train set
y_test_encoded = label_encoder.transform(y_test)  # use on test set

print("Encoded has done!")


# Get the mapping of original labels to encoded labels
label_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))

print("Label mapping:", label_mapping)

y_test


y_test_encoded

# Integer label to One-hot conversion
y_train_one_hot = to_categorical(y_train_encoded, num_classes=len(label_encoder.classes_))
y_test_one_hot = to_categorical(y_test_encoded, num_classes=len(label_encoder.classes_))

print("One-hot conversion has done!")

y_test_encoded

y_test_one_hot

# Esure all the sets have same formate
print('Train image: ', type(X_train_augmented))
print('Train label: ', type(y_train_one_hot))
print('Test image: ',  type(X_test))
print('Train label: ', type(y_test_one_hot))

# Shapes of each sets
X_train_augmented.shape, y_train_one_hot.shape, X_test.shape, y_test_one_hot.shape

# Load the RestNet152 as base model with out top
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze the base model layers
for layer in base_model.layers:
    layer.trainable = True


# Custome Top Dense layers
# x = Flatten()(base_model.output)
x = GlobalAveragePooling2D()(base_model.output)
# x = BatchNormalization()(x)
# x = Dense(1024, activation='relu')(x)
# x = BatchNormalization()(x)
x = Dropout(0.4)(x)
output = Dense(4, activation='softmax')(x)  # Since our number of class is 4

# Create the final model
model = Model(inputs=base_model.input, outputs=output)

# Compiles the model
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# Model summary
# model.summary()

# plot the model architecture:
# plot_model(model, to_file="Model_Architecture.png", show_shapes=True, show_layer_names=True)


# Early stoping
# early_stop = EarlyStopping(monitor ='val_loss', patience=5, restore_best_weights=True,verbose=1)
reduce_lr = ReduceLROnPlateau(monitor ='val_loss', factor=0.3, patience=2, min_lr=1e-10,verbose=1)

# Add ModelCheckpoint to save the best model based on validation loss
model_checkpoint = ModelCheckpoint('Best_Model_On_Partial.h5', monitor='val_loss', save_best_only=True, mode='min', verbose=1)

# fits the model on our dataset
history = model.fit(X_train_augmented, y_train_one_hot,
                    epochs = 50,
                    batch_size = 32,
                    validation_data=(X_test, y_test_one_hot),
                    verbose = 1,
                    callbacks=[reduce_lr, model_checkpoint]
)

Predicted_model = load_model('/Best_Model_On_Partial.h5')  # Load your saved model
print(f'Trained model Best_Model.h5 has loaded')

X_test.shape

# predictions on test set
batch_size = 32  # Adjust batch size based on your GPU memory capacity
pred_lab_dist = []
for i in range(0, len(X_test), batch_size):
    batch = X_test[i:i+batch_size]
    batch_pred = Predicted_model.predict(batch,verbose=0)
    pred_lab_dist.append(batch_pred)

# Concatenate all batch predictions
pred_lab_dist = np.concatenate(pred_lab_dist, axis=0)

# true label
Y_true = y_test_encoded
# predicted  label
Y_pred = np.argmax(pred_lab_dist, axis=-1)

# Classification report
cls_rep = classification_report(Y_true,Y_pred)
print(f'Classification report:\n {cls_rep}')

# Confusion matrix
cnf_mtx = confusion_matrix(Y_true,Y_pred)
print(f'Confusion matrix:\n {cnf_mtx}')

# figure size
plt.figure(figsize=(15,6))

# Training accuracy and validation accuracy
train_acc = history.history['accuracy']
val_acc   = history.history['val_accuracy']
epochs = range(1,20) # as best performance at epoch 19
plt.subplot(1,2,1)
plt.plot(epochs,train_acc[:19],label='Train Accuracy')
plt.plot(epochs,val_acc[:19],label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuarcy')
plt.title('Accuracy Plot')
plt.legend(loc='center right')

# Training and validation losses
train_loss = history.history['loss']
val_loss   = history.history['val_loss']
plt.subplot(1,2,2)
plt.plot(epochs,train_loss[:19],label='Train Loss')
plt.plot(epochs,val_loss[:19],label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss Plot')
plt.legend(loc='upper right')

plt.show()

# Plot confusion matrix
plt.figure(figsize=(7, 6))
sns.heatmap(cnf_mtx, annot=True, fmt='d', cmap='Blues', xticklabels=img_labels, yticklabels=img_labels)
plt.xlabel('Predicted Label')
plt.xticks(rotation=45)
plt.ylabel('True Label')
plt.yticks(rotation=0)
plt.title('Confusion Matrix')
plt.show()

# Function to perform complete image processing:
def Image_PreProcessed(img,crop=None,target_size=(224,224)):
    img = img.copy()
    if crop is not None:
        _,_,img = crop(img)
    # resize
    img = cv2.resize(img, dsize=target_size, interpolation=cv2.INTER_LANCZOS4)
    # filtering
    img = cv2.bilateralFilter(img, 2, 50,50)
    # apply colormap bone for highlight important features
    img = cv2.applyColorMap(img, cv2.COLORMAP_BONE)
    # Normalize
    img = (img/255.0).astype('float32')

    # return processed image
    return img

    # Read images
img_dir = "/kaggle/input/brain-tumor-mri-dataset/Testing"
# Image labels
img_labels = ['glioma','meningioma','notumor','pituitary']
# Image paths
img_folders = [(os.path.join(img_dir,lab),lab) for lab in img_labels] # (folder,labels) pairs


img_folders

# Extract 5 images from each of types
images = [(os.listdir(img_folder)[:5],lab,img_folder) for img_folder,lab in img_folders]
print(images)

# predicts the tumor types on each unseen data
# plot figure
# Calculate rows and columns dynamically
n_images = 5*len(images) * 2  # Original  + predicting
cols = 4
rows = (n_images + cols - 1) // cols  # Round up to fit all images
plt.figure(figsize=(20,rows*4))
i = 1
for imgs,lab,folder in images:
    original_label = lab
    for img in imgs:
        # read image
        image_path = os.path.join(folder,img)
        image = cv2.imread(image_path)
        image = cv2.resize(image, (224,224))  # Resize to the expected input size
        # Original image
        plt.subplot(rows,cols,i)
        plt.imshow(image,cmap='gray')
        plt.title(f'Original:{original_label}')
        plt.axis('off')

        # Predicted type with score on processed image
        image_pro = Image_PreProcessed(image,crop=Crop_image)
        # Dynamically reshape , since target size is (1,224,224,3), 1 for batch size for single image
        target_shape = (1, *image_pro.shape)  # (1, height, width, channels)
        image_pro_re = image_pro.reshape(target_shape)
        predictions = Predicted_model.predict(image_pro_re,verbose=0)
        predicted_label = np.argmax(predictions, axis=-1)[0]  # Get the index of the highest probability
        predicted_score = np.max(predictions,axis=-1)[0]
        plt.subplot(rows,cols,i+1)
        plt.imshow(image_pro ,cmap='gray')
        plt.title(f'Prediction:{img_labels[predicted_label]}\nScore:{np.round(predicted_score*100,2)}%')
        plt.axis('off')
        i+=2
plt.show()
