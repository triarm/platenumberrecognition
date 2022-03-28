import os
import albumentations as A

annotations_dir = 'Dataset/annotations/'
masks_dir = 'Dataset/masks'
images_dir = 'Dataset/images/'


# Open JSON file
with open('Dataset/annotations/annotate_motor.json', 'r') as f:
    data_motor = json.load(f)

with open('Dataset/annotations/annotate_mobil.json', 'r') as f:
    data_mobil = json.load(f)


# Create mask from annotation Label Studio
for annotation in data_motor:
    points = annotation['annotations'][0]['result'][0]['value']['points']
    height = annotation['annotations'][0]['result'][0]['original_height']
    width = annotation['annotations'][0]['result'][0]['original_width']
    file_name = os.path.splitext(annotation['file_upload'])[0]

    # # KTP area based on image
    area = [[int(point[0]*width/100), int(point[1]*height/100)] for point in points]

    blank_image = np.zeros(shape=[height, width], dtype=np.uint8)
    cv2.fillPoly(blank_image, pts = [np.array(area)], color =255)

    cv2.imwrite(f'{masks_dir}/{file_name}.png', blank_image)
    
for annotation in data_mobil:
    points = annotation['annotations'][0]['result'][0]['value']['points']
    height = annotation['annotations'][0]['result'][0]['original_height']
    width = annotation['annotations'][0]['result'][0]['original_width']
    file_name = os.path.splitext(annotation['file_upload'])[0]

    # # KTP area based on image
    area = [[int(point[0]*width/100), int(point[1]*height/100)] for point in points]

    blank_image = np.zeros(shape=[height, width], dtype=np.uint8)
    cv2.fillPoly(blank_image, pts = [np.array(area)], color =255)

    cv2.imwrite(f'{masks_dir}/{file_name}.png', blank_image)


# Augmentation

images_augmentation_dir = 'Dataset/augmentation/images/'
masks_augmentation_dir = 'Dataset/augmentation/masks/'

## List Transformation
transform = A.Compose([
                       A.ShiftScaleRotate(shift_limit=0.1,
                                          rotate_limit=15,
                                          scale_limit=0.1,
                                          p=0.5),
                       A.RandomBrightnessContrast(brightness_limit=0.2,
                                                  contrast_limit=0.2,
                                                  p=0.5),
                       A.Resize(height=256,
                                width=256,
                                interpolation=cv2.INTER_LANCZOS4)
                    ])

## Augmentation Process
for img in os.listdir(images_dir):
    for id in range(5):
        try:
            filename = os.path.splitext(img)[0]
            # Open image and mask and convert to array
            image_ori = cv2.imread(f"{images_dir}/{img}")
            mask_ori = cv2.imread(f"{masks_dir}/{filename}.png")

            #transform image
            transformed = transform(image=image_ori, mask=mask_ori)
            transformed_image = transformed['image']
            transformed_mask = transformed['mask']

            # Save image
            cv2.imwrite(f"{images_augmentation_dir}/{filename}_{id+1}.png", transformed_image)
            cv2.imwrite(f"{masks_augmentation_dir}/{filename}_{id+1}.png", transformed_mask)
        except:
            print(f"{img} error to be augmentation")

