# PixelPerfekt-Toolkit
---
## Project Description: 📷✨
PixelPerfekt Toolkit is a web application built with Streamlit, offering a range of image processing functionalities to transform pixels with precision. Users can upload images in JPG, JPEG, or PNG format and apply various operations such as converting to grayscale, viewing in HSV color space, applying color masks, detecting contours, and applying different types of image blurring filters. The toolkit provides interactive sliders and checkboxes for parameter adjustment and visualization of histograms to understand pixel distributions.

## Video Demonstration 📹
Click Below to watch the video 👇👇👇

<a href="https://www.loom.com/share/6252e15c1c5648c78b9bbf0ec7bdc4f4?sid=bff091b1-cfcc-4f2b-977e-ca4e12b306c1" target="_blank">
<img width="991" alt="Screenshot 2024-08-18 at 07 13 39" src="https://github.com/user-attachments/assets/b2db011b-1dbd-4552-9d9a-c34200e8ff7e">
</a>

## Features: 🌟
- Upload images in JPG, JPEG, or PNG format.
- Convert images to grayscale.
- View images in HSV color space.
- Apply color masks based on user-defined color ranges.
- Detect contours in the image.
- Sharpen images using a user-specified alpha and beta values.
- Visualize histograms for both original and processed images in REALTIME.
- Option to view enlarged histograms.

## Folder Structure: 🗂️

The application organizes the output images into respective folders based on the applied filter:

- **Original_Image/**: Contains original images and their histograms.
- **Gray/**: Stores grayscale images and their histograms.
- **Box_filter/**: Saves images processed with the box filter and their histograms.
- **Gaussian_blur/**: Stores Gaussian-blurred images and corresponding histograms.
- **Median_blur/**: Contains images processed with median blur and their histograms.
- **Bilateral_filter/**: Stores images processed with bilateral filtering and their histograms.
- **Kernel_blur/**: Saves images processed with kernel blur and their histograms.
- **Sharpened/**: Stores sharpened images and their histograms.
- **Color_mask/**: Contains color-masked images and corresponding histograms.
- **Contours/**: Stores contour-detected images and their histograms.
- **Hsv/**: Saves images in HSV color space and corresponding histograms.

## Docker Integration: 🐳
To streamline the setup and avoid recreating virtual environments, the application is containerized using Docker. This ensures consistent execution across different environments.

## Filters Implemented: ⚡️
1. `Original` (Not technically a filter)
2. `Gray`
3. `HSV`
4. `Color Mask`
5. `Contours`
6. `Kernel Blur`
7. `Box Filter`
8. `Gaussian Blur`
9. `Median Blur`
10. `Bilateral Filter`
11. `Sharpened`

## Technologies Used: 🛠️

- Streamlit
- OpenCV
- NumPy
- Matplotlib
- Python Imaging Library (PIL)
- Docker

### Running the Application with Docker:
- **Build and run the Docker Image:**
   ```bash
   docker-compose up --build
   ```
- **Access the Application:**
Open your web browser and navigate to
```bash
http://localhost:8501
```

**📌 Note: For optimal performance, it's recommended to run the application on a machine with sufficient computational resources, especially when processing large images or applying complex operations.**
