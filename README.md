# Image to Pencil Sketch

## Assignment
Select an image in **RGB format** and use the **OpenCV Python library** to transform it so that it resembles a **pencil sketch**. 

### Tips
- Convert the RGB image to **grayscale** — this will turn the image into a classic black-and-white photo. 
- **Invert** the grayscale image — sometimes referred to as a negative image, useful for enhancing details.
- Mix the grayscale image with the inverted **blurry** image — by dividing pixel values of the grayscale image by the inverted blurry image; the result should resemble a pencil sketch.
- Experiment with other OpenCV transformations to improve the sketch effect.

## Data Description
We are providing you with a **sample image of a dog**, but you can choose any colored image you want to complete this project.

## Practicalities
Make sure that the solution **reflects your entire thought process** — how your code is structured and the reasoning behind your approach is more important than the final files.

## Methodology

This project converts a color (RGB) image into a pencil-sketch-style image using OpenCV. The overall approach is based on a classic image processing technique that enhances edges and suppresses smooth regions to mimic hand-drawn pencil strokes.

### Step 1: Grayscale Conversion
First, the input color image is converted to grayscale. This step removes color information and reduces the problem to intensity analysis only, which aligns with the nature of pencil sketches that are primarily defined by light and shadow rather than color.

### Step 2: Image Inversion
Next, the grayscale image is inverted. Inversion transforms dark regions into light ones and vice versa, which is useful for emphasizing edges after blending. This step prepares the image for the dodge blending operation used later in the pipeline.

### Step 3: Gaussian Blurring
The inverted grayscale image is then blurred using a Gaussian filter. Blurring smooths out fine-grained noise and creates a soft background representation of the image. The kernel size controls how strong this smoothing effect is: larger kernels produce softer, more abstract sketches, while smaller kernels preserve more detail.

### Step 4: Dodge Blending
After blurring, the pencil sketch effect is created using a dodge blend operation between the original grayscale image and the blurred inverted image. Specifically, pixel values from the grayscale image are divided by the inverted blurred image. This operation brightens regions with strong edges while keeping flat regions light, producing an effect similar to pencil shading on paper.

### Step 5: Post-processing
Finally, optional post-processing steps such as sharpening or thresholding can be applied. Sharpening enhances line clarity, while thresholding can produce a more stylized, line-art-like sketch. These steps allow for experimentation and fine-tuning of the visual result without changing the core logic of the algorithm.

### Summary
Overall, the pipeline is designed to be simple, interpretable, and modular. Each step serves a clear purpose in transforming the original image into a pencil sketch, and intermediate outputs can be inspected to better understand how the final result is formed.
