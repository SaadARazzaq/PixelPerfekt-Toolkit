import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
from PIL import Image

hsv_img = None
kernel_size = (25, 25)

st.set_page_config(page_title='PixelPerfekt Toolkit', page_icon='logo.jpg')
st.title("PixelPerfekt Toolkit 📷✨")
st.subheader("Transforming Pixels with Precision")

noisy_img = st.file_uploader("Upload Image", type=['jpg', 'jpeg', 'png'])

if noisy_img is not None:
    file_content = noisy_img.read()
    np_array = np.frombuffer(file_content, np.uint8)
    decode_img = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
    decode_img = cv2.cvtColor(decode_img, cv2.COLOR_BGR2RGB)
    hsv_img = cv2.cvtColor(decode_img, cv2.COLOR_RGB2HSV)

    selected_operation = st.sidebar.selectbox("Select Operation", ["Original", "Gray", "HSV", "Color Mask", "Contours",
                                                                  "Kernel Blur", "Box Filter", "Gaussian Blur",
                                                                  "Median Blur", "Bilateral Filter", "Sharpened"])
    
    if selected_operation == "Gray":
        gray_img = cv2.cvtColor(decode_img, cv2.COLOR_RGB2GRAY)

        col1, col2 = st.columns(2)
        col1.image(decode_img, caption="Original Image", use_column_width=True)
        col2.image(gray_img, caption="Gray Image", use_column_width=True)

        col_original, col_gray = st.columns(2)

        hist_original, bins_original = np.histogram(decode_img.flatten(), bins=256, range=[0, 256])
        fig_original, ax_original = plt.subplots(figsize=(8, 6))
        ax_original.hist(decode_img.flatten(), bins=256, range=[0, 256], color='blue', alpha=0.7)
        ax_original.set_title('Original Image Histogram')
        ax_original.set_xlabel('Pixel Value')
        ax_original.set_ylabel('Frequency')
        hist_img_path_original = 'hist_img_original.png'
        fig_original.savefig(hist_img_path_original, format='png', bbox_inches='tight')
        plt.close(fig_original)
        hist_img_original = Image.open(hist_img_path_original)
        col_original.image(hist_img_original, caption="Original Image Histogram", width=350)

        hist_gray, bins_gray = np.histogram(gray_img.flatten(), bins=256, range=[0, 256])
        fig_gray, ax_gray = plt.subplots(figsize=(8, 6))
        ax_gray.hist(gray_img.flatten(), bins=256, range=[0, 256], color='gray', alpha=0.7)
        ax_gray.set_title('Gray Image Histogram')
        ax_gray.set_xlabel('Pixel Value')
        ax_gray.set_ylabel('Frequency')
        hist_img_path_gray = 'hist_img_gray.png'
        fig_gray.savefig(hist_img_path_gray, format='png', bbox_inches='tight')
        plt.close(fig_gray)
        hist_img_gray = Image.open(hist_img_path_gray)
        col_gray.image(hist_img_gray, caption="Gray Image Histogram", width=350)

        if st.checkbox("Show Enlarged Histogram"):
                st.image(hist_img_gray, caption="Enlarged Histogram", use_column_width=True)


    elif selected_operation == "HSV":
        col1, col2 = st.columns(2)
        col1.image(decode_img, caption="Original Image", use_column_width=True)
        col2.image(hsv_img, caption="HSV Image", use_column_width=True)

        col_original, col_hsv = st.columns(2)

        hist_original, bins_original = np.histogram(decode_img.flatten(), bins=256, range=[0, 256])
        fig_original, ax_original = plt.subplots(figsize=(8, 6))
        ax_original.hist(decode_img.flatten(), bins=256, range=[0, 256], color='blue', alpha=0.7)
        ax_original.set_title('Original Image Histogram')
        ax_original.set_xlabel('Pixel Value')
        ax_original.set_ylabel('Frequency')
        hist_img_path_original = 'hist_img_original.png'
        fig_original.savefig(hist_img_path_original, format='png', bbox_inches='tight')
        plt.close(fig_original)
        hist_img_original = Image.open(hist_img_path_original)
        col_original.image(hist_img_original, caption="Original Image Histogram", width=350)

        hist_hsv, bins_hsv = np.histogram(hsv_img.flatten(), bins=256, range=[0, 256])
        fig_hsv, ax_hsv = plt.subplots(figsize=(8, 6))
        ax_hsv.hist(hsv_img.flatten(), bins=256, range=[0, 256], color='green', alpha=0.7)
        ax_hsv.set_title('HSV Image Histogram')
        ax_hsv.set_xlabel('Pixel Value')
        ax_hsv.set_ylabel('Frequency')
        hist_img_path_hsv = 'hist_img_hsv.png'
        fig_hsv.savefig(hist_img_path_hsv, format='png', bbox_inches='tight')
        plt.close(fig_hsv)
        hist_img_hsv = Image.open(hist_img_path_hsv)
        col_hsv.image(hist_img_hsv, caption="HSV Image Histogram", width=350)

        if st.checkbox("Show Enlarged Histogram"):
            st.image(hist_img_hsv, caption="Enlarged Histogram", use_column_width=True)

    elif selected_operation == "Color Mask":
        low_h = st.sidebar.slider("Low Hue", 0, 255, 0)
        high_h = st.sidebar.slider("High Hue", 0, 255, 255)
        low_s = st.sidebar.slider("Low Saturation", 0, 255, 0)
        high_s = st.sidebar.slider("High Saturation", 0, 255, 255)
        low_v = st.sidebar.slider("Low Value", 0, 255, 0)
        high_v = st.sidebar.slider("High Value", 0, 255, 255)

        lower_color_range = np.array([low_h, low_s, low_v])
        upper_color_range = np.array([high_h, high_s, high_v])

        mask_img = cv2.inRange(hsv_img, lower_color_range, upper_color_range)

        col1, col2 = st.columns(2)
        col1.image(decode_img, caption="Original Image", use_column_width=True)
        col2.image(mask_img, caption="Masked Image", use_column_width=True)

        col_original, col_mask = st.columns(2)

        hist_original, bins_original = np.histogram(decode_img.flatten(), bins=256, range=[0, 256])
        fig_original, ax_original = plt.subplots(figsize=(8, 6))
        ax_original.hist(decode_img.flatten(), bins=256, range=[0, 256], color='blue', alpha=0.7)
        ax_original.set_title('Original Image Histogram')
        ax_original.set_xlabel('Pixel Value')
        ax_original.set_ylabel('Frequency')
        hist_img_path_original = 'hist_img_original.png'
        fig_original.savefig(hist_img_path_original, format='png', bbox_inches='tight')
        plt.close(fig_original)
        hist_img_original = Image.open(hist_img_path_original)
        col_original.image(hist_img_original, caption="Original Image Histogram", width=350)

        hist_masked, bins_masked = np.histogram(mask_img.flatten(), bins=256, range=[0, 256])
        fig_masked, ax_masked = plt.subplots(figsize=(8, 6))
        ax_masked.hist(mask_img.flatten(), bins=256, range=[0, 256], color='orange', alpha=0.7)
        ax_masked.set_title('Masked Image Histogram')
        ax_masked.set_xlabel('Pixel Value')
        ax_masked.set_ylabel('Frequency')
        hist_img_path_masked = 'hist_img_masked.png'
        fig_masked.savefig(hist_img_path_masked, format='png', bbox_inches='tight')
        plt.close(fig_masked)
        hist_img_masked = Image.open(hist_img_path_masked)
        col_mask.image(hist_img_masked, caption="Masked Image Histogram", width=350)

        if st.checkbox("Show Enlarged Histogram"):
            st.image(hist_img_masked, caption="Enlarged Histogram", use_column_width=True)


    elif selected_operation == "Contours":
        if hsv_img is None:
            st.warning("Please select 'HSV' operation first to generate the color mask.")
        else:
            low_h = st.sidebar.slider("Low Hue", 0, 255, 0)
            high_h = st.sidebar.slider("High Hue", 0, 255, 255)
            low_s = st.sidebar.slider("Low Saturation", 0, 255, 0)
            high_s = st.sidebar.slider("High Saturation", 0, 255, 255)
            low_v = st.sidebar.slider("Low Value", 0, 255, 0)
            high_v = st.sidebar.slider("High Value", 0, 255, 255)

            lower_color_range = np.array([low_h, low_s, low_v])
            upper_color_range = np.array([high_h, high_s, high_v])

            mask_img = cv2.inRange(hsv_img, lower_color_range, upper_color_range)
            contours, _ = cv2.findContours(mask_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contour_img = decode_img.copy()
            cv2.drawContours(contour_img, contours, -1, (0, 255, 0), 2)

            col1, col2 = st.columns(2)
            col1.image(decode_img, caption="Original Image", use_column_width=True)
            col2.image(contour_img, caption="Image with Contours", use_column_width=True)

            col_original, col_contours = st.columns(2)

            hist_original, bins_original = np.histogram(decode_img.flatten(), bins=256, range=[0, 256])
            fig_original, ax_original = plt.subplots(figsize=(8, 6))
            ax_original.hist(decode_img.flatten(), bins=256, range=[0, 256], color='blue', alpha=0.7)
            ax_original.set_title('Original Image Histogram')
            ax_original.set_xlabel('Pixel Value')
            ax_original.set_ylabel('Frequency')
            hist_img_path_original = 'hist_img_original.png'
            fig_original.savefig(hist_img_path_original, format='png', bbox_inches='tight')
            plt.close(fig_original)
            hist_img_original = Image.open(hist_img_path_original)
            col_original.image(hist_img_original, caption="Original Image Histogram", width=350)

            hist_contour, bins_contour = np.histogram(contour_img.flatten(), bins=256, range=[0, 256])
            fig_contour, ax_contour = plt.subplots(figsize=(8, 6))
            ax_contour.hist(contour_img.flatten(), bins=256, range=[0, 256], color='purple', alpha=0.7)
            ax_contour.set_title('Image with Contours Histogram')
            ax_contour.set_xlabel('Pixel Value')
            ax_contour.set_ylabel('Frequency')
            hist_img_path_contour = 'hist_img_contour.png'
            fig_contour.savefig(hist_img_path_contour, format='png', bbox_inches='tight')
            plt.close(fig_contour)
            hist_img_contour = Image.open(hist_img_path_contour)
            col_contours.image(hist_img_contour, caption="Image with Contours Histogram", width=350)

            if st.checkbox("Show Enlarged Histogram"):
                st.image(hist_img_contour, caption="Enlarged Histogram", use_column_width=True)


    elif selected_operation == "Kernel Blur":
        kernel_size = st.sidebar.slider("Kernel Size", 1, 50, 25)
        kernel_size = (kernel_size, kernel_size)  # Convert to tuple
        kernel_value = np.ones(kernel_size, np.float32) / (kernel_size[0] * kernel_size[1])
        output_kernel = cv2.filter2D(decode_img, -1, kernel_value, borderType=cv2.BORDER_CONSTANT)

        col1, col2 = st.columns(2)
        col1.image(decode_img, caption="Original Image", use_column_width=True)
        col2.image(output_kernel, caption="Kernel Blur Image", use_column_width=True)

        col_original, col_k_Blur = st.columns(2)

        hist_original, bins_original = np.histogram(decode_img.flatten(), bins=256, range=[0, 256])
        fig_original, ax_original = plt.subplots(figsize=(8, 6))
        ax_original.hist(decode_img.flatten(), bins=256, range=[0, 256], color='blue', alpha=0.7)
        ax_original.set_title('Original Image Histogram')
        ax_original.set_xlabel('Pixel Value')
        ax_original.set_ylabel('Frequency')
        hist_img_path_original = 'hist_img_original.png'
        fig_original.savefig(hist_img_path_original, format='png', bbox_inches='tight')
        plt.close(fig_original)
        hist_img_original = Image.open(hist_img_path_original)
        col_original.image(hist_img_original, caption="Original Image Histogram", width=350)

        hist_kernel, bins_kernel = np.histogram(output_kernel.flatten(), bins=256, range=[0, 256])
        fig_k_Blur, ax_k_Blur = plt.subplots(figsize=(8, 6))
        ax_k_Blur.hist(output_kernel.flatten(), bins=256, range=[0, 256], color='orange', alpha=0.7)
        ax_k_Blur.set_title('Kernel Blur Image Histogram')
        ax_k_Blur.set_xlabel('Pixel Value')
        ax_k_Blur.set_ylabel('Frequency')
        hist_img_path_k_Blur = 'hist_img_k_Blur.png'
        fig_k_Blur.savefig(hist_img_path_k_Blur, format='png', bbox_inches='tight')
        plt.close(fig_k_Blur)
        hist_img_k_Blur = Image.open(hist_img_path_k_Blur)
        col_k_Blur.image(hist_img_k_Blur, caption="Kernel Blur Image Histogram", width=350)

        if st.checkbox("Show Enlarged Histogram"):
            st.image(hist_img_k_Blur, caption="Enlarged Histogram", use_column_width=True)


    elif selected_operation == "Box Filter":
        kernel_size = st.sidebar.slider("Kernel Size", 1, 50, 25)
        kernel_size = (kernel_size, kernel_size)  # Convert to tuple
        box = cv2.boxFilter(decode_img, -1, kernel_size, normalize=True)

        col1, col2 = st.columns(2)
        col1.image(decode_img, caption="Original Image", use_column_width=True)
        col2.image(box, caption="Box Filter Image", use_column_width=True)

        col_original, col_box_Blur = st.columns(2)

        hist_original, bins_original = np.histogram(decode_img.flatten(), bins=256, range=[0, 256])
        fig_original, ax_original = plt.subplots(figsize=(8, 6))
        ax_original.hist(decode_img.flatten(), bins=256, range=[0, 256], color='blue', alpha=0.7)
        ax_original.set_title('Original Image Histogram')
        ax_original.set_xlabel('Pixel Value')
        ax_original.set_ylabel('Frequency')
        hist_img_path_original = 'hist_img_original.png'
        fig_original.savefig(hist_img_path_original, format='png', bbox_inches='tight')
        plt.close(fig_original)
        hist_img_original = Image.open(hist_img_path_original)
        col_original.image(hist_img_original, caption="Original Image Histogram", width=350)

        hist_box, bins_box = np.histogram(box.flatten(), bins=256, range=[0, 256])
        fig_box_Blur, ax_box_Blur = plt.subplots(figsize=(8, 6))
        ax_box_Blur.hist(box.flatten(), bins=256, range=[0, 256], color='green', alpha=0.7)
        ax_box_Blur.set_title('Box Filter Image Histogram')
        ax_box_Blur.set_xlabel('Pixel Value')
        ax_box_Blur.set_ylabel('Frequency')
        hist_img_path_box_Blur = 'hist_img_box_Blur.png'
        fig_box_Blur.savefig(hist_img_path_box_Blur, format='png', bbox_inches='tight')
        plt.close(fig_box_Blur)
        hist_img_box_Blur = Image.open(hist_img_path_box_Blur)
        col_box_Blur.image(hist_img_box_Blur, caption="Box Filter Image Histogram", width=350)

        if st.checkbox("Show Enlarged Histogram"):
            st.image(hist_img_box_Blur, caption="Enlarged Histogram", use_column_width=True)


    elif selected_operation == 'Gaussian Blur':
        kernel_size = st.sidebar.slider("Kernel Size", 1, 50, 25)
        # Ensure the kernel size is odd and greater than 0
        kernel_size = (kernel_size + 1) if kernel_size % 2 == 0 else kernel_size
        gaussian_image = cv2.GaussianBlur(decode_img, (kernel_size, kernel_size), 0)

        col1, col2 = st.columns(2)
        col1.image(decode_img, caption="Original Image", use_column_width=True)
        col2.image(gaussian_image, caption="Gaussian Blur Image", use_column_width=True)

        col_original, col_g_Blur = st.columns(2)

        hist_original, bins_original = np.histogram(decode_img.flatten(), bins=256, range=[0, 256])
        fig_original, ax_original = plt.subplots(figsize=(8, 6))
        ax_original.hist(decode_img.flatten(), bins=256, range=[0, 256], color='blue', alpha=0.7)
        ax_original.set_title('Original Image Histogram')
        ax_original.set_xlabel('Pixel Value')
        ax_original.set_ylabel('Frequency')
        hist_img_path_original = 'hist_img_original.png'
        fig_original.savefig(hist_img_path_original, format='png', bbox_inches='tight')
        plt.close(fig_original)
        hist_img_original = Image.open(hist_img_path_original)
        col_original.image(hist_img_original, caption="Original Image Histogram", width=350)

        hist_gaussian, bins_gaussian = np.histogram(gaussian_image.flatten(), bins=256, range=[0, 256])
        fig_g_Blur, ax_g_Blur = plt.subplots(figsize=(8, 6))
        ax_g_Blur.hist(gaussian_image.flatten(), bins=256, range=[0, 256], color='purple', alpha=0.7)
        ax_g_Blur.set_title('Gaussian Blur Image Histogram')
        ax_g_Blur.set_xlabel('Pixel Value')
        ax_g_Blur.set_ylabel('Frequency')
        hist_img_path_g_Blur = 'hist_img_g_Blur.png'
        fig_g_Blur.savefig(hist_img_path_g_Blur, format='png', bbox_inches='tight')
        plt.close(fig_g_Blur)
        hist_img_g_Blur = Image.open(hist_img_path_g_Blur)
        col_g_Blur.image(hist_img_g_Blur, caption="Gaussian Blur Image Histogram", width=350)

        if st.checkbox("Show Enlarged Histogram"):
            st.image(hist_img_g_Blur, caption="Enlarged Histogram", use_column_width=True)


    elif selected_operation == 'Median Blur':
        kernel_size = st.sidebar.slider("Kernel Size", 1, 50, 5)
        # Ensure kernel size is odd
        kernel_size = kernel_size + 1 if kernel_size % 2 == 0 else kernel_size
        median_image = cv2.medianBlur(decode_img, kernel_size)

        col1, col2 = st.columns(2)
        col1.image(decode_img, caption="Original Image", use_column_width=True)
        col2.image(median_image, caption="Median Blur Image / Noise Reduction", use_column_width=True)

        col_original, col_m_Blur = st.columns(2)

        hist_original, bins_original = np.histogram(decode_img.flatten(), bins=256, range=[0, 256])
        fig_original, ax_original = plt.subplots(figsize=(8, 6))
        ax_original.hist(decode_img.flatten(), bins=256, range=[0, 256], color='blue', alpha=0.7, edgecolor='black')
        ax_original.set_title('Original Image Histogram')
        ax_original.set_xlabel('Pixel Value')
        ax_original.set_ylabel('Frequency')

        hist_img_path_original = 'hist_img_original.png'
        fig_original.savefig(hist_img_path_original, format='png', bbox_inches='tight')
        plt.close(fig_original)

        hist_img_original = Image.open(hist_img_path_original)
        col_original.image(hist_img_original, caption="Original Image Histogram", width=350)

        hist_m_Blur, bins_m_Blur = np.histogram(median_image.flatten(), bins=256, range=[0, 256])
        fig_m_Blur, ax_m_Blur = plt.subplots(figsize=(8, 6))
        ax_m_Blur.hist(median_image.flatten(), bins=256, range=[0, 256], color='green', alpha=0.7, edgecolor='black')
        ax_m_Blur.set_title('Median Blur Image Histogram')
        ax_m_Blur.set_xlabel('Pixel Value')
        ax_m_Blur.set_ylabel('Frequency')

        hist_img_path_m_Blur = 'hist_img_m_Blur.png'
        fig_m_Blur.savefig(hist_img_path_m_Blur, format='png', bbox_inches='tight')
        plt.close(fig_m_Blur)

        hist_img_m_Blur = Image.open(hist_img_path_m_Blur)
        col_m_Blur.image(hist_img_m_Blur, caption="Median Blur Image Histogram", width=350)
        
        if st.checkbox("Show Enlarged Histogram"):
            st.image(hist_img_m_Blur, caption="Enlarged Histogram", use_column_width=True)

    elif selected_operation == 'Bilateral Filter':
        bilateral_image = cv2.bilateralFilter(decode_img, 5, 6, 6)  # Applying bilateral filter with a 5x5 neighborhood, color similarity sigma=6, and spatial proximity sigma=6

        col1, col2 = st.columns(2)
        col1.image(decode_img, caption="Original Image", use_column_width=True)
        col2.image(bilateral_image, caption="Bilateral Filter Image", use_column_width=True)

        col_original, col_bil_Blur = st.columns(2)

        hist_original, bins_original = np.histogram(decode_img.flatten(), bins=256, range=[0, 256])
        fig_original, ax_original = plt.subplots(figsize=(8, 6))
        ax_original.plot(bins_original[:-1], hist_original, color='blue', alpha=0.7)
        ax_original.set_title('Original Image Histogram')
        ax_original.set_xlabel('Pixel Value')
        ax_original.set_ylabel('Frequency')
        hist_img_path_original = 'hist_img_original.png'
        fig_original.savefig(hist_img_path_original, format='png', bbox_inches='tight')
        plt.close(fig_original)
        hist_img_original = Image.open(hist_img_path_original)
        col_original.image(hist_img_original, caption="Original Image Histogram", width=350)

        hist_bilateral, bins_bilateral = np.histogram(bilateral_image.flatten(), bins=256, range=[0, 256])
        fig_bilateral, ax_bilateral = plt.subplots(figsize=(8, 6))
        ax_bilateral.plot(bins_bilateral[:-1], hist_bilateral, color='purple', alpha=0.7)
        ax_bilateral.set_title('Bilateral Filter Image Histogram')
        ax_bilateral.set_xlabel('Pixel Value')
        ax_bilateral.set_ylabel('Frequency')
        hist_img_path_bilateral = 'hist_img_bilateral.png'
        fig_bilateral.savefig(hist_img_path_bilateral, format='png', bbox_inches='tight')
        plt.close(fig_bilateral)
        hist_img_bilateral = Image.open(hist_img_path_bilateral)
        col_bil_Blur.image(hist_img_bilateral, caption="Bilateral Filter Image Histogram", width=350)
        if st.checkbox("Show Enlarged Histogram"):
                st.image(hist_img_path_bilateral, caption="Enlarged Histogram", use_column_width=True)


    elif selected_operation.startswith("Sharpened"):
        alpha = st.sidebar.slider("Alpha Values", 0, 10)
        beta = st.sidebar.slider("Beta Values", 0, 10)

        if alpha is not None and beta is not None:
            gaussian_blur = cv2.GaussianBlur(decode_img, (7, 7), 2)
            sharpened_image = cv2.addWeighted(decode_img, alpha, gaussian_blur, -beta, 0)

            col1, col2 = st.columns(2)
            col1.image(decode_img, caption="Original Image", use_column_width=True)
            col2.image(sharpened_image, caption=f"Sharpened {alpha} {beta}", use_column_width=True)

            col_original, col_sharpened = st.columns(2)

            hist_original, bins_original = np.histogram(decode_img.flatten(), bins=256, range=[0, 256])
            fig_original, ax_original = plt.subplots(figsize=(8, 6))
            ax_original.hist(decode_img.flatten(), bins=256, range=[0, 256], color='blue', alpha=0.7)
            ax_original.set_title('Original Image Histogram')
            ax_original.set_xlabel('Pixel Value')
            ax_original.set_ylabel('Frequency')
            fig_original.savefig('hist_original.png', format='png', bbox_inches='tight')
            plt.close(fig_original)

            hist_img_original = Image.open('hist_original.png')
            col_original.image(hist_img_original, caption="Original Image Histogram", width=350)

            hist_sharpened, bins_sharpened = np.histogram(sharpened_image.flatten(), bins=256, range=[0, 256])
            fig_sharpened, ax_sharpened = plt.subplots(figsize=(8, 6))
            ax_sharpened.hist(sharpened_image.flatten(), bins=256, range=[0, 256], color='red', alpha=0.7)
            ax_sharpened.set_title(f'Sharpened Image Histogram (Alpha={alpha}, Beta={beta})')
            ax_sharpened.set_xlabel('Pixel Value')
            ax_sharpened.set_ylabel('Frequency')
            fig_sharpened.savefig('hist_sharpened.png', format='png', bbox_inches='tight')
            plt.close(fig_sharpened)

            hist_img_sharpened = Image.open('hist_sharpened.png')
            col_sharpened.image(hist_img_sharpened, caption=f'Sharpened Image Histogram (Alpha={alpha}, Beta={beta})', width=350)

            if st.checkbox("Show Enlarged Histogram"):
                st.image(hist_img_sharpened, caption="Enlarged Histogram", use_column_width=True)

    else:
        st.image(decode_img, caption="Original Image", width=400)

        hist_original, bins_original = np.histogram(decode_img.flatten(), bins=256, range=[0, 256])
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.hist(decode_img.flatten(), bins=256, range=[0, 256], color='blue', alpha=0.7)
        ax.set_title('Original Image Histogram')
        ax.set_xlabel('Pixel Value')
        ax.set_ylabel('Frequency')
        
        
        hist_img_path = 'hist_img.png'
        fig.savefig(hist_img_path, format='png', bbox_inches='tight') # Save the histogram figure as an image
        plt.close(fig)
        
        
        hist_img = Image.open(hist_img_path) # Display the saved histogram image
        st.image(hist_img, caption="Original Image Histogram", width=400)

        if st.checkbox("Show Enlarged Histogram"):
            st.image(hist_img, caption="Enlarged Histogram", use_column_width=True)