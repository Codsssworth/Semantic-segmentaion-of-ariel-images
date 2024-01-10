#
import random
import os
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import cv2
from tensorflow import  keras
os.environ["SM_FRAMEWORK"] = "tf.keras"
import itertools
from matplotlib.patches import Patch

import segmentation_models as sm
from training import  X_test, y_train, y_test
from keras.metrics import MeanIoU
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, f1_score, accuracy_score,auc
from skimage.segmentation import find_boundaries
from simple_multi_unet_model import multi_unet_model, jacard_coef
from keras.models import load_model

def intersection_over_union(y_true, y_pred, class_label):

    intersection = np.logical_and(y_true == class_label, y_pred == class_label).sum()
    union = np.logical_or(y_true == class_label, y_pred == class_label).sum()

    # Handle the case of division by zero
    if union == 0:
        return 1.0

    iou = intersection / union
    return iou

patch_size=256
weights = [0.1666, 0.1666, 0.1666, 0.1666, 0.1666, 0.1666]
dice_loss = sm.losses.DiceLoss(class_weights=weights)
focal_loss = sm.losses.CategoricalFocalLoss()
total_loss = dice_loss + (1 * focal_loss)  #
# Register the custom loss function
losses = {
    'dice_loss': dice_loss,
    'focal_loss': focal_loss,
    'dice_loss_plus_1focal_loss': total_loss,
    'jacard_coef':jacard_coef
}

# Load the model with the custom loss function
model = load_model("models/sim.hdf5", custom_objects=losses)

#IOU

y_pred=model.predict(X_test)
y_pred_argmax=np.argmax(y_pred, axis=3)
y_test_argmax=np.argmax(y_test, axis=3)


#Using built in keras function for IoU

n_classes = 6
IOU_keras = MeanIoU(num_classes=n_classes)
IOU_keras.update_state(y_test_argmax, y_pred_argmax)
print("Mean IoU =", IOU_keras.result().numpy())

# #######################################################################
# #Predict on a few images

f1_scores = []
accuracy_scores = []
miou_scores = []
boundary_f1_scores=[]
all_test_images = []
all_ground_truths = []
all_predicted_images = []
x=10

for i in range(0,x):
    test_img_number = random.randint(0, len(X_test))
    test_img = X_test[test_img_number]
    ground_truth=y_test_argmax[test_img_number]
    #test_img_norm=test_img[:,:,0][:,:,None]
    test_img_input=np.expand_dims(test_img,axis = 0)
    prediction = (model.predict(test_img_input))
    predicted_img=np.argmax(prediction, axis=3)[0,:,:]

    all_test_images.append( test_img )
    all_ground_truths.append( ground_truth )
    all_predicted_images.append( predicted_img )

    # Flatten the ground_truth and prediction arrays
    ground_truth_flat = ground_truth.flatten()
    prediction_flat = prediction.flatten()

    # Compute F1 score
    f1 = f1_score( ground_truth_flat, predicted_img.flatten(), average='weighted' )
    f1_scores.append( f1 )

    # Compute miou
    miou = np.mean( [intersection_over_union( ground_truth, predicted_img, c ) for c in range( n_classes )] )
    miou_scores.append( miou )

    # Compute Accuracy
    accuracy = accuracy_score( ground_truth_flat, predicted_img.flatten() )
    accuracy_scores.append( accuracy )

    # Compute Boundary F1 Score
    boundary_ground_truth = find_boundaries(ground_truth, mode='inner')
    boundary_predicted = find_boundaries(predicted_img, mode='inner')
    boundary_f1 = f1_score(boundary_ground_truth.flatten(), boundary_predicted.flatten())
    boundary_f1_scores.append(boundary_f1)



num_images_per_plot = 3
num_plots = (x + num_images_per_plot - 1) // num_images_per_plot
num_plots = min(num_plots, 4)  # Limit the number of plots to 4 (3 images per plot)

plt.figure(figsize=(16, 10))
for i, images in enumerate(itertools.zip_longest(all_test_images, all_ground_truths, all_predicted_images, fillvalue=None)):
    if i >= num_plots:
        break

    plt.subplot(num_plots, 3, i*3 + 1)
    plt.title('Testing Image')
    if images[0] is not None:
        plt.imshow(images[0])
    plt.axis('off')

    plt.subplot(num_plots, 3, i*3 + 2)
    plt.title('Testing Label')
    if images[1] is not None:
        plt.imshow(images[1])
    plt.axis('off')

    plt.subplot(num_plots, 3, i*3 + 3)
    plt.title('Prediction on test image')
    if images[2] is not None:
        plt.imshow(images[2])
    plt.axis('off')

plt.tight_layout()
legend_elements = [
    Patch( facecolor="#50E3C2", edgecolor='k', label='Water' ),
    Patch( facecolor="#F5A623", edgecolor='k', label='Land (unpaved area)' ),
    Patch( facecolor="#DE597F", edgecolor='k', label='Road' ),
    Patch( facecolor="#D0021B", edgecolor='k', label='Building' ),
    Patch( facecolor="#417505", edgecolor='k', label='Vegetation' ),
    Patch( facecolor="#9B9B9B", edgecolor='k', label='Unlabeled' )
]
plt.legend( handles=legend_elements, loc='lower left', bbox_to_anchor=(1, 1), title="Legend" )
plt.show()
# Plot the evaluation curves
plt.figure(figsize=(18, 12))

# Subplot for F1 Score
plt.subplot(231)
plt.title('F1 Score')
plt.plot(f1_scores, 'b')
plt.axhline(y=np.mean(f1_scores), color='r', linestyle='--')
plt.text(len(f1_scores)-1, np.mean(f1_scores), f'Avg: {np.mean(f1_scores):.4f}', color='r', ha='right')
plt.xlabel('Test Images')
plt.ylabel('F1 Score')

# Subplot for Mean Intersection over Union (mIoU)
plt.subplot(232)
plt.title('Mean Intersection over Union (mIoU)')
plt.plot(miou_scores, 'g')
plt.axhline(y=np.mean(miou_scores), color='r', linestyle='--')
plt.text(len(miou_scores)-1, np.mean(miou_scores), f'Avg: {np.mean(miou_scores):.4f}', color='r', ha='right')
plt.xlabel('Test Images')
plt.ylabel('mIoU')

# Subplot for Accuracy
plt.subplot(233)
plt.title('Accuracy')
plt.plot(accuracy_scores, 'r')
plt.axhline(y=np.mean(accuracy_scores), color='r', linestyle='--')
plt.text(len(accuracy_scores)-1, np.mean(accuracy_scores), f'Avg: {np.mean(accuracy_scores):.4f}', color='r', ha='right')
plt.xlabel('Test Images')
plt.ylabel('Accuracy')

# Subplot for Boundary F1 Score
plt.subplot(234)
plt.title('Boundary F1 Score')
plt.plot(boundary_f1_scores, 'm')
plt.axhline(y=np.mean(boundary_f1_scores), color='r', linestyle='--')
plt.text(len(boundary_f1_scores)-1, np.mean(boundary_f1_scores), f'Avg: {np.mean(boundary_f1_scores):.4f}', color='r', ha='right')
plt.xlabel('Test Images')
plt.ylabel('Boundary F1 Score')

plt.suptitle('Semantic Segmentation Model Evaluation')  # Main title for the figure
plt.tight_layout()
plt.show()


#########################################
#unseen images
dir_path = 'test'
image_files = os.listdir(dir_path)
images=[]
pred=[]


for file_name in image_files:
    # Load the image and normalize its pixel values
    file_path = os.path.join( dir_path, file_name )
    img = cv2.imread( file_path, 1 )
    img = cv2.resize( img, (256, 256) )
    SIZE_X = (img.shape[1] // patch_size) * patch_size
    SIZE_Y = (img.shape[0] // patch_size) * patch_size
    img = Image.fromarray( img )
    img = img.crop( (0, 0, SIZE_X, SIZE_Y) )
    org = img
    # image = image.resize((SIZE_X, SIZE_Y))  #Try not to resize for semantic segmentation
    img = np.array( img )
    img=np.expand_dims( img, axis=0 )
    img=img / 255.0
    predictions = (model.predict( img ))
    prediction_mask = np.argmax( predictions, axis=-1 )

    plt.figure( figsize=(12, 8) )

    plt.subplot( 231 )
    plt.imshow( org )
    plt.title( 'Image' )
    plt.axis( 'off' )

    plt.subplot( 232 )
    plt.imshow( prediction_mask[0], cmap='jet', alpha=0.7 )
    plt.title( 'Predicted Mask' )
    plt.axis( 'off' )

    plt.suptitle( 'Prediction over User Input Images', fontsize=16 )
    plt.tight_layout( rect=[0, 0.03, 1, 0.95] )
    # Add legends for colors
    legend_elements = [
        Patch( facecolor="#50E3C2", edgecolor='k', label='Water' ),
        Patch( facecolor="#F5A623", edgecolor='k', label='Land (unpaved area)' ),
        Patch( facecolor="#DE597F", edgecolor='k', label='Road' ),
        Patch( facecolor="#D0021B", edgecolor='k', label='Building' ),
        Patch( facecolor="#417505", edgecolor='k', label='Vegetation' ),
        Patch( facecolor="#9B9B9B", edgecolor='k', label='Unlabeled' )
    ]
    plt.legend( handles=legend_elements, loc='lower left', bbox_to_anchor=(1, 1), title="Legend" )
    plt.show()

