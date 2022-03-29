# Objective
1. Create ML model for plate segmentation
2. Create algorithm for segment the character on plate
3. Create ML model for recognize character
4. Create algoritm for processing the process

# Dataset
- [Labelled Indonesian Car and Plate](https://www.kaggle.com/gunawan26/labelled-indonesian-car-and-plate-number)
- [Indonesian Plate Number](https://www.kaggle.com/datasets/imamdigmi/indonesian-plate-number)
- [OLX](https://www.olx.co.id)

# Segmentation Model
We will use UNET architecture for doing plate segmentation. In this project, we doing three architecture modification.
1. UNET architecture
2. UNET architecture with reduced the number of filter
3. UNET architecture by using VGG19 weights on encoder

# References
- Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net: Convolutional Networks for Biomedical Image Segmentation. (N. Navab, J. Hornegger, W. M. Wells, & A. F. Frangi, Penyunt.) *Medical Image Computing and Computer-Assisted Intervention -- MICCAI 2015*, 234-241.
- Simonyan, K., & Zisserman, A. (2015). *Very Deep Convolutional Networks for Large-Scale Image Recognition*. 
