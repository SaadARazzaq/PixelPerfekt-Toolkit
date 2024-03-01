# PixelPerfekt-Toolkit
---
## Project Description: 📷✨
PixelPerfekt Toolkit is a web application built with Streamlit, offering a range of image processing functionalities to transform pixels with precision. Users can upload images in JPG, JPEG, or PNG format and apply various operations such as converting to grayscale, viewing in HSV color space, applying color masks, detecting contours, and applying different types of image blurring filters. The toolkit provides interactive sliders and checkboxes for parameter adjustment and visualization of histograms to understand pixel distributions.

## Features: 🌟
- Upload images in JPG, JPEG, or PNG format.
- Convert images to grayscale.
- View images in HSV color space.
- Apply color masks based on user-defined color ranges.
- Detect contours in the image.
- Apply various image blurring filters including kernel blur, box filter, Gaussian blur, median blur, and bilateral filter.
- Sharpen images using a user-specified alpha and beta values.
- Visualize histograms for both original and processed images.
- Option to view enlarged histograms.

## Usage: 🖥️
1. Upload an image by clicking the "Upload Image" button.
2. Select the desired operation from the sidebar dropdown menu.
3. Adjust parameters using sliders or checkboxes, if applicable.
4. View the processed image and its corresponding histogram.
5. Optionally, check the "Show Enlarged Histogram" checkbox to view the histogram in a larger size.

## Technologies Used: 🛠️
- Streamlit
- OpenCV
- NumPy
- Matplotlib
- Python Imaging Library (PIL)

## How to Run: ▶️
1. Install the required libraries by running `pip install streamlit opencv-python-headless numpy matplotlib pillow`.
2. Save the provided code in a file named `app.py`.
3. Open a terminal and navigate to the directory containing `app.py`.
4. Run the Streamlit application by executing `streamlit run app.py`.
5. Access the application in a web browser at `http://localhost:8501`.

## Future Enhancements: 🔮
- Add more image processing operations such as edge detection, morphological operations, and image transformations.
- Improve the user interface with additional customization options and interactive elements.
- Optimize image processing algorithms for better performance with large images.

📌 **Note: For optimal performance, it's recommended to run the application on a machine with sufficient computational resources, especially when processing large images or applying complex operations.**
