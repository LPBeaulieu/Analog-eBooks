import copy
from datetime import datetime
import glob
import json
import math
import numpy as np
import os
import pymupdf
import re
import shutil
import signal
import sys
import textwrap
import tempfile
import traceback
import time


#The "clear_screen()" function will clear the CLI screen
#using the appropriate command depending on the operating system.
def clear_screen():
    #'nt' is for Windows, 'posix is for Linux/Raspberry Pi/macOS (else statement)
    os.system('cls' if os.name == 'nt' else 'clear')


#The Signal Interrupt (SIGINT) handler will
#call the "signal_interrupt_signal_handler()" function 
#when the user presses on CTRL + C to exit the app.

#The function "signal_interrupt_signal_handler()" will call
#"sys.exit(0)" to exit the program normally.
def signal_interrupt_signal_handler(sig, frame):
    sys.exit(0)


#The function "write_entry_in_error_log()" will write 
#the full technical traceback error to the error log.
def write_entry_in_error_log():
    with open("ERROR LOG.txt", "a", encoding="utf-8") as error_log:
        error_log.write(f"\n--- Error at {datetime.now()} ---\n")
        traceback.print_exc(file=error_log)


#The function "get_list_average_value()" will return the
#average value of a list of digits, provided that the list
#isn't empty, in which case it will return "None".
def get_list_average_value(list_of_digits):
    length_of_list_of_digits = len(list_of_digits)
    if length_of_list_of_digits > 0: 
        return round(sum(list_of_digits)/length_of_list_of_digits)
    else:
        return None


#The "process_image()" function will extract the image file from the
#PDF document as a grayscale Pixmap object, convert it to a NumPy array 
#to process it, and then convert the NumPy array back to a Pixmap object,
#which will be included in the "doc_output" Document object.
def process_image(  doc,
                    doc_output,
                    page_index,
                    dpi_setting, 
                    cumulative_pdf_file_size_estimation,                   
                    do_filter_out_splotches_margins,
                    number_of_standard_deviations_for_filtering_page_color_cropping,
                    number_of_standard_deviations_for_filtering_page_color,
                    number_of_standard_deviations_for_filtering_splotches_margins,
                    do_filter_out_splotches_entire_page,
                    number_of_standard_deviations_for_filtering_splotches_entire_page,
                    black_and_white_mode_enabled,
                    do_crop_pages,                
                    horizontal_crop_kernel_size_height_percent,
                    horizontal_crop_kernel_radius_kernel_size_percent,
                    horizontal_crop_margin_buffer_width_percentage,
                    vertical_crop_kernel_size_height_percent,
                    vertical_crop_kernel_radius_kernel_size_percent,
                    vertical_crop_margin_buffer_height_percentage,
                    brightness_level,
                    final_brightness_level,
                    contrast_level,
                    final_contrast_level,
                    is_dark_mode_enabled,
                    left_margin_width_percent,
                    right_margin_width_percent,
                    top_margin_height_percent,
                    bottom_margin_height_percent,
                    list_of_cropped_page_widths,
                    set_of_potential_blank_pages,
                    list_of_original_document_page_numbers
                  ):
    #The current page of the PDF document
    #is stored in the "page" variable.
    page = doc[page_index]

    #A Grayscale Pixmap object with "csGRAY" colorspace
    #and the provided "dpi_setting" is created from the
    #"page" object.
    pixmap = page.get_pixmap(colorspace=pymupdf.csGRAY, dpi=dpi_setting)

    #The "pixmap" Pixmap object is converted to a NumPy array for filtering out 
    #yellow pixels (since it is a grayscale pixmap, the number of color channels "pixmap.n" == 1):
    img_array_view = np.frombuffer(pixmap.samples_mv, dtype=np.uint8).reshape(pixmap.h, pixmap.w, pixmap.n)

    #As the grayscale NumPy array was obtained through PyMuPDF,
    #It has the following shape: (Height, Width, 1), which needs
    #to be squeezed into a (Height, Width) 2D array with the 
    #"np.squeeze()" method before doing any transformations
    #on the data.
    img_array_view = np.squeeze(img_array_view)

    #The "img_array_view" NumPy array view is converted to 
    #float32 and normalized in the range to 0.0-1.0 for statistical analysis
    img_array_view = img_array_view.astype(np.float32)/255.0

    #A copy is made of the image array view, as modifications will be made to it.
    img_array = img_array_view.copy()

    img_array_cropping = img_array_view.copy()

    #The initial mean and standard deviation of all grayscale pixel values
    #will be used to filter out the paper color pixels.
    initial_mean_pixel_value = np.mean(img_array)
    initial_pixel_value_standard_deviation = np.std(img_array)

    #The pixels that are lighter than the mean value of all pixels found 
    #on the page before any modifications are made to the numpy array, plus 
    #"number_of_standard_deviations_for_filtering_page_color" times the standard
    #deviation will be set to white (value of one). You may need to use a negative
    #value for "number_of_standard_deviations_for_filtering_page_color" if the pages
    #have significant yellowing around the edges, or if there is a shadow left behind
    #by the spine. This will then filter out these pixels more aggressively.
    img_array[img_array > initial_mean_pixel_value + number_of_standard_deviations_for_filtering_page_color * initial_pixel_value_standard_deviation] = 1

    img_array_cropping[img_array_cropping > initial_mean_pixel_value + number_of_standard_deviations_for_filtering_page_color_cropping * initial_pixel_value_standard_deviation] = 1

    #The brightness of the pixels is adjusted by multiplying all of the pixel values
    #by the "brightness_level". Black pixels (zeroes) will be left untouched, as 
    #anything multiplied by zero gives zero.
    if brightness_level != 1 and brightness_level > 0:
        img_array *= brightness_level
        #Clip values to ensure they stay between zero and 1
        #(in case the "brightness_level" value is greater than
        #1 and the initial pixel value is greater than 0.5)
        img_array = np.clip(img_array, 0.0, 1.0)

        #The initial mean of all grayscale pixel values needs
        #to be adjusted in the same way as the pixels were 
        #brightened, in order to adjust the baseline average
        #grayscale value that will be used when adjusting
        #the contrast
        initial_mean_pixel_value *= brightness_level
        if initial_mean_pixel_value > 1.0:
            initial_mean_pixel_value = 1.0

    ##The initial height and width of
    #the NumPy "img_array" are stored
    #in the "height" and "width" variables,
    #respectively. 
    height = img_array.shape[0]
    width = img_array.shape[1]

    #The slice coordinates for the pixels at the center of the page 
    #(excluding those from the left, right, top and bottom margins)
    #are stored in the variable "slice_center_of_page" through the
    #use of the NumPy "np.s_" indexing routine.
    slice_center_of_page = np.s_[
        round(top_margin_height_percent * height) : round(height - bottom_margin_height_percent * height),
        round(left_margin_width_percent * width) : round(width - right_margin_width_percent * width)
    ]
    #The NumPy view of the center pixels obtained by slicing
    #"img_array_cropping" with the "slice_center_of_page" object is stored
    #in the variable "center_of_page" variable.
    center_of_page = img_array_cropping[slice_center_of_page]

    #The mean and standard deviation of all non-white pixels on the page will
    #be used when filtering out the remaining blotches on the outer edges of 
    #the page (if "do_filter_out_splotches_margins == True").
    non_white_pixels_in_center_of_page = center_of_page[center_of_page != 1]

    #A threshold of 30 non-white pixels in the center of the page is used to meet the minimal
    #sample size for a normal distribution and obtain a reliable mean and standard deviation
    #of a non-blank page (see the "if" below).
    number_of_non_white_pixels_in_center_of_page = np.sum(non_white_pixels_in_center_of_page)

    #The potentially blank page index (+1 as it is zero-indexed)
    #is added to the set "set_of_potential_blank_pages", as the
    #total count of non-white pixels in the center of the page is
    #inferior to 30 (cutoff point for a blank page).

    #A threshold of 30 non-white pixels in the center of the page is used to meet the minimal
    #sample size for a normal distribution and obtain a reliable mean and standard deviation
    #of a non-blank page.
    if (number_of_non_white_pixels_in_center_of_page < 30):    
        set_of_potential_blank_pages.add(page_index + 1)
    #If the "do_filter_out_splotches" mode is enabled and the pixels at the center of the page
    #aren't all white ("not np.all(non_white_pixels_in_center_of_page == 1)"), the "if" statement
    #below will filter out the lighter gray pixels in the margins of the page.

    #A threshold of 30 non-white pixels in the center of the page is used to meet the minimal
    #sample size for a normal distribution and obtain a reliable mean and standard deviation
    #of a non-blank page.
    elif (do_filter_out_splotches_margins or do_filter_out_splotches_entire_page or contrast_level != 1):

        #The mean and standard deviation of all non-white pixels on the page will
        #be used when filtering out the remaining blotches on the outer edges of 
        #the page (if "do_filter_out_splotches_margins == True").
        mean_non_white_pixel_value = np.mean(non_white_pixels_in_center_of_page)
        standard_deviation_non_white_pixel_value = np.std(non_white_pixels_in_center_of_page)

        #The margins filter will only affect pixels lighter than the threshold
        #of "mean_non_white_pixel_value + number_of_standard_deviations_for_filtering_splotches_margins * 
        #standard_deviation_non_white_pixel_value" in the NumPy array that is used for cropping 
        #("img_array_cropping"), as the margins need to be clear of yellowing or shadows before
        #the cropping step. This step will not affect the contents of the page themselves, as
        #the final pages will contain the center content from the lightly filtered "img_array",
        #and the margins of the heavily filtered "img_array_cropping".

        #The "if" statement below will only run if at least one of the 
        #margin percentages is greater than zero.
        if (do_filter_out_splotches_margins and 
            not(
                top_margin_height_percent <= 0 and 
                bottom_margin_height_percent <= 0 and
                left_margin_width_percent <= 0 and
                right_margin_width_percent <=0 
                )
            ):
            #Pixels in the page ("img_array_cropping") that are lighter than the
            #threshold calculated from the mean grayscale value of the 
            #non-white pixels in the center of the page, plus 
            #"number_of_standard_deviations_for_filtering_splotches_margins"
            #times the standard deviation of these pixels.
            pixels_lighter_than_threshold = (img_array_cropping > mean_non_white_pixel_value + 
                number_of_standard_deviations_for_filtering_splotches_margins * standard_deviation_non_white_pixel_value)

            #Select everything except the center of the page by
            #creating a Boolean mask initialized to "True" (1) in 
            #every position.
            spatial_mask_margins_of_page = np.ones(img_array_cropping.shape, dtype = bool)
            #Set the values of the indices in the center of the page
            #to "False".
            spatial_mask_margins_of_page[slice_center_of_page] = False
            #The spatial and value mask will select for pixels in the margins that are
            #lighter than the threshold
            spatial_and_value_mask = spatial_mask_margins_of_page & pixels_lighter_than_threshold
            #The pixels in the margins of the page that are lighter than
            #the threshold will be converted to white pixels ("1"), which
            #will help remove remaining blotches stemming from heavy yellowing
            #of the pages.
            img_array_cropping[spatial_and_value_mask] = 1        
        #The contrast level needs to be adjusted before filtering out all remaining blotches on the page,
        #as these should roughly be about the same grayscale value as the baseline for the initial page
        #before any changes were made ("initial_mean_pixel_value"), and so be minimally affected by the
        #contrast operation, while the text will become much darker. This will make it easier to filter
        #out the blotches, without losing much of the text information in the process. The formula for 
        #the contrast adjustment was taken form the "Pil.Image.blend()" Pillow method that is used when
        #adjusting the contrast with Pillow's ImageEnhance module ("Pil.ImageEnhance.Contrast()").

        #If we were to use the updated value of the mean grayscale pixel value across all of the page
        #once the "yellow" pixels have been filtered out (replaced with "1" for white), the resulting 
        #updated mean would be much lighter (i.e., close to 1) than the original background color of 
        #the page (approximately the color of which the remaining blotches are) and the blotches would
        #be darkened along with the text, making them more difficult to filter out. This is why the
        #"initial_mean_pixel_value" and not the updated mean value after filtering out the "yellow" 
        #pixels needs to be used in the contrast code below.
        if (contrast_level != 1 and contrast_level >= 0):
            img_array = img_array * contrast_level + initial_mean_pixel_value * (1.0 - contrast_level)
            #Clip values to ensure they stay between zero and 1
            img_array = np.clip(img_array, 0.0, 1.0)

            #As contrast was adjusted, we need to recalculate
            #the mean non white pixel value and the standard deviation.
            #As there are likely more white pixels after the contrast step,
            #we need to re-slice the center of the page to select non-white pixels.

            #The mean and standard deviation of all non-white pixels on the page will
            #be used when filtering out the remaining blotches on the outer edges of 
            #the page (if "do_filter_out_splotches_margins == True").
            non_white_pixels_in_center_of_page = center_of_page[center_of_page != 1]
            mean_non_white_pixel_value = np.mean(non_white_pixels_in_center_of_page)
            standard_deviation_non_white_pixel_value = np.std(non_white_pixels_in_center_of_page)

        #If the filter is applied to all of the page for the pixels
        #that are lighter than the threshold, the "elif" statement
        #below will run.
        if do_filter_out_splotches_entire_page:
            #Pixels in the page ("img_array") that are lighter than the
            #threshold calculated from the mean grayscale value of the 
            #non-white pixels in the center of the page, plus 
            #"number_of_standard_deviations_for_filtering_splotches_entire_page"
            #times the standard deviation of these pixels.
            pixels_lighter_than_threshold = (img_array > mean_non_white_pixel_value + 
                number_of_standard_deviations_for_filtering_splotches_entire_page * standard_deviation_non_white_pixel_value)
            img_array[pixels_lighter_than_threshold] = 1

        #Any pixels that are almost white (0.95/1 or 242/255)
        #will be changed to white to avoid darkening them 
        #in the final brightness adjustment (which will
        #be used to darken the interior of the letters).
        img_array[img_array > 0.95] = 1
        #As contrast was adjusted and near-white pixels were changed to white, 
        #we need to recalculate the mean non white pixel value and the standard deviation.

        #As there are likely more white pixels after the "img_array[img_array > 0.95] = 1"
        #operation, we need to re-slice the center of the page to select non-white pixels.

        #The mean and standard deviation of all non-white pixels on the page will
        #be used when filtering out the remaining blotches on the outer edges of 
        #the page (if "do_filter_out_splotches_margins == True").
        non_white_pixels_in_center_of_page = center_of_page[center_of_page != 1]
        mean_non_white_pixel_value_final_brightness_adjustment = np.mean(non_white_pixels_in_center_of_page)
        standard_deviation_non_white_pixel_value_final_brightness_adjustment = np.std(non_white_pixels_in_center_of_page)

        #The interior of the letters will be selectively darkened by multiplying their grayscale pixel value by a value between
        #zero and one, where darker pixels (those farther to the left of the mean in the bell curve) will be darkened more so
        #than pixels closer to the mean or to the right of the mean, so as to avoid darkening the anti-aliasing pixels. 
        #The modifier "final_brightness_percentage_adjustment" may be used to fine-tune this final brightness adjustment.

        #Each category (except the first one) is comprised of two boundaries, where the pixels need to be smaller than the 
        #right bound and greater or equal to the left bound (e.g., "img_array[greater_or_equal_to_minus_50 & lesser_than_minus_25]").
        #Each of the bounds are represented below as a percentage of the standard deviation (e.g. "lesser_than_minus_50" means smaller
        #than minus 50% of the standard deviation to the left of the mean ("mean - 0.5 * standard deviation")).
        lesser_than_minus_50 = img_array < mean_non_white_pixel_value_final_brightness_adjustment - 0.50 * standard_deviation_non_white_pixel_value_final_brightness_adjustment

        greater_or_equal_to_minus_50 = img_array >= mean_non_white_pixel_value_final_brightness_adjustment - 0.50 * standard_deviation_non_white_pixel_value_final_brightness_adjustment

        lesser_than_minus_25 = img_array < mean_non_white_pixel_value_final_brightness_adjustment - 0.25 * standard_deviation_non_white_pixel_value_final_brightness_adjustment

        greater_or_equal_to_minus_25 = img_array >= mean_non_white_pixel_value_final_brightness_adjustment - 0.25 * standard_deviation_non_white_pixel_value_final_brightness_adjustment

        lesser_than_0 = img_array < mean_non_white_pixel_value_final_brightness_adjustment

        greater_or_equal_to_0 = img_array >= mean_non_white_pixel_value_final_brightness_adjustment

        lesser_than_plus_25 = img_array < mean_non_white_pixel_value_final_brightness_adjustment + 0.25 * standard_deviation_non_white_pixel_value_final_brightness_adjustment

        greater_or_equal_to_plus_25 = img_array >= mean_non_white_pixel_value_final_brightness_adjustment + 0.25 * standard_deviation_non_white_pixel_value_final_brightness_adjustment

        lesser_than_plus_50 = img_array < mean_non_white_pixel_value_final_brightness_adjustment + 0.50 * standard_deviation_non_white_pixel_value_final_brightness_adjustment

        greater_or_equal_to_plus_50 = img_array >= mean_non_white_pixel_value_final_brightness_adjustment + 0.50 * standard_deviation_non_white_pixel_value_final_brightness_adjustment

        lesser_than_plus_75 = img_array < mean_non_white_pixel_value_final_brightness_adjustment + 0.75 * standard_deviation_non_white_pixel_value_final_brightness_adjustment

        greater_or_equal_to_plus_75 = img_array >= mean_non_white_pixel_value_final_brightness_adjustment + 0.75 * standard_deviation_non_white_pixel_value_final_brightness_adjustment

        lesser_than_plus_100 = img_array < mean_non_white_pixel_value_final_brightness_adjustment + 1.00 * standard_deviation_non_white_pixel_value_final_brightness_adjustment

        #The user would input how many times they want the text to be darker 
        #(e.g., 2.0 times darker would give 1.0/2.0 = 0.5 as the value of 
        #"inverse_final_brightness_level", instead of the default value of 1).
        inverse_final_brightness_level = 1
        #In the case where "final_brightness_level" is zero,
        #we want to avoid a "divide by zero" error.
        if final_brightness_level > 0:
            inverse_final_brightness_level = 1 / final_brightness_level

        img_array[lesser_than_minus_50] = 0.025 * inverse_final_brightness_level

        img_array[greater_or_equal_to_minus_50 & lesser_than_minus_25] *= 0.05 * inverse_final_brightness_level

        img_array[greater_or_equal_to_minus_25 & lesser_than_0] *= 0.10 * inverse_final_brightness_level

        img_array[greater_or_equal_to_0 & lesser_than_plus_25] *= 0.20 * inverse_final_brightness_level

        img_array[greater_or_equal_to_plus_25 & lesser_than_plus_50] *= 0.30 * inverse_final_brightness_level

        img_array[greater_or_equal_to_plus_50 & lesser_than_plus_75] *= 0.40 * inverse_final_brightness_level

        img_array[greater_or_equal_to_plus_75 & lesser_than_plus_100] *= 0.50 * inverse_final_brightness_level

        #Clip values to ensure they stay between zero and 1
        img_array = np.clip(img_array, 0.0, 1.0)

        #The "final_contrast_level" adjusts the contrast once the filters and the final brightness level
        #have been applied. The same "initial_mean_pixel_value" that was used for the first contrast step
        #will be used once again (because at this point most pixels are white ("1") so the average pixel 
        #value for all pixels would be very close to white and all other pixels would be darkened).
        if (final_contrast_level != 1 and final_contrast_level >= 0):
            img_array = img_array * final_contrast_level + initial_mean_pixel_value * (1.0 - final_contrast_level)
            #Clip values to ensure they stay between zero and 1
            img_array = np.clip(img_array, 0.0, 1.0)

        #Any pixels that are less than five percent away from being
        #pure black are set to zero (full black).
        img_array[img_array < 0.05] = 0

    #If the pages are to be set to black and white and cropped,
    #The page color pixels that were initially white (1s) will
    #be set to 0s and other pixels will be set to 1s so that 
    #the non-white pixels can be summed up for the convolution
    #step.
    if do_crop_pages:

        #The center of the original grayscale image "img_array" that was lightly filtered so as to preserve the anti-aliasing pixels
        #will be "pasted" onto the heavily filtered image array before the inversion of the polarity ("img_array_cropping_before_inversion"),
        #in order to ensure that the margins are nice and clean, as both a heavy full page initial filter and a stringent margins filter
        #were applied to "img_array_cropping".
        img_array_cropping_before_inversion = img_array_cropping.copy()

        #The black and white NumPy array consisting of 0s for black pixels
        #and 1s for white pixels will be inverted so that the black pixels
        #can be added up (adding the 1s) in order to detect the edges of 
        #the text.
        img_array_cropping = np.where(img_array_cropping != 1, 1, 0)

        #A NumPy convolution operation will be performed in order to detect
        #contiguous horizontal pixels belonging to the block of text. The kernel
        #is comprised of a 1D NumPy array (1D as we are traversing a 1D array of 
        #black and white pixels) initialized with ones, and its values will be 
        #incremented each times it crosses black pixels (1s) in the inverted page 
        #NumPy array. Larger kernel sizes will be able to "see" pixels across larger 
        #gaps, but might result in the inclusion of more noise. It is a balancing act, 
        #but generally, a kernel size of about 2% of the initial page height should 
        #give reasonable results in detecting contiguous columns of black pixels,
        #resulting from the addition of all black pixels along all rows for these
        #given columns. 
        horizontal_crop_kernel_size = round(horizontal_crop_kernel_size_height_percent * height)
        #The kernel radius, expressed as a percentage of the kernel size, determines 
        #what overlap of black pixels within the kernel is required in order for them to be 
        #blurred together in the convolution step. The maximum value for this is the kernel 
        #size itself (complete overlap, or 100% kernel size), which wouldn't allow for any 
        #white pixels (gaps). A value of around 30% the kernel size is usually good for 
        #detecting contiguous columns of black pixels (detecting the left and right edges 
        #of the text), while around 20% of the kernel size is used when detecting contiguous 
        #rows of black pixels (detecting the top and bottom edges of the text). A smaller 
        #threshold is used in the latter case because there can be empty lines in-between
        #paragraphs, or larger vertical spaces between the end of a chapter and the beginning
        #of the next chapter. 

        #If you need to cover greater vertical gaps when detecting the top and bottom edges 
        #of the page, you will likely need to both increase the kernel size from its initial 
        #value of 8% of the initial height of the page, and potentially decrease the kernel 
        #threshold from its value of 20% of the adjusted kernel size.
        horizontal_crop_kernel_threshold = round(horizontal_crop_kernel_radius_kernel_size_percent * horizontal_crop_kernel_size)

        #The rows are added for each column of "img_array_cropping",
        #(hence the "axis = 0"), in order to tally up the
        #number of black pixels (1s) in each column. This 
        #will allow to check for contiguity when determining
        #the horizontal margins of the text block.  
        col_sums = np.sum(img_array_cropping, axis=0)
        #The maximum value of "col_sums"1D horizontal array
        #will allow to determine if the page is likely a blank page
        #("maximum_sum <= height * 0.02") or a page that
        #contains content otherwise ("if" statement below).
        maximum_sum = np.max(col_sums)
        #If contiguous non-white pixels within the threshold of the kernel radius were detected horizontally and vertically
        #("horizontal_cropping_successful == True and vertical_cropping_successful == True"), then the NumPy array will be cropped.
        horizontal_cropping_successful = False
        #A minimum threshold of black pixels added along all rows for each column ("col_sums")
        #of two percent of the initial page height +5 is used as a threshold to say that there
        #is content in that given column. 
        if maximum_sum > height * 0.02:
            #+5 added in case the value of np.max(col_sums) is very small, as in a blank page.
            #This ensures that absolute silence (0 pixels) never triggers as the "content"

            #The "flatten()" method will collapse the array to a 1D array, as the kernel is also 1D
            #and will check for contiguity across each column of black pixels (within the bounds of 
            #"horizontal_crop_kernel_threshold").
            has_content = (col_sums > 0.01 * height + 5).astype(np.uint8).flatten()
            kernel = np.ones(horizontal_crop_kernel_size, dtype=np.uint8)
            #The parameter "mode='same'" will ensure that the output array from the convolution
            #step is the exact same length as the input image width. This will allow to map where
            #the left and right edges of the image are in the original image, based on the output layer's
            #first and last indices that meet the threshold requirement.             
            try:
                #The "horizontal_crop_kernel_threshold" is the minimum number of "hits" in a kernel window 
                #(typically 30% of the kernel size) to call it a block of text.
                smoothed = np.convolve(has_content, kernel, mode='same') >= horizontal_crop_kernel_threshold
            except Exception as e:
                #The function "get_terminal_dimensions()" will return the number of columns 
                #and rows in the console, to allow to properly format the text and dividers.
                columns, lines = get_terminal_dimensions()
                error_string = textwrap.fill("Please either increase the value of 'Left-Right Crop Kernel Size Percentage' and/or 'Left-Right Crop Kernel Radius Percentage', as no contiguous black pixels were detected during the horizontal convolution step when cropping the left and right margins of the pages.", width=columns)

                print("\n" + "=" * columns)
                print("CRITICAL ERRROR ENCOUNTERED")
                print("\nDetails:", e)
                print("\n" + "=" * error_string)

                #The function "write_entry_in_error_log()" will write 
                #the full technical traceback error to the error log.
                write_entry_in_error_log()

                sys.exit(error_string + "\n")

            #When cropping the page horizontally, a margin trimming 
            #buffer will expand the crop selection by a number of 
            #pixels proportional to the original page, to ensure 
            #that no text is lost, and to preserve some kind of 
            #margin for the block of text.

            #This means that it should be relatively safe to make all
            #pixels within these buffers white (zero) in "smoothed",
            #as they will be added back anyways, provided that some
            #text is detected right after the buffers (which should 
            #be the case when the book is formatted with only one
            #column of text). Doing this will help to crop out some
            #of the shadows cast by the spine in the books.
            if horizontal_crop_margin_buffer_width_percentage > 0:
                automatic_margin_trimming_buffer = round(horizontal_crop_margin_buffer_width_percentage * width)
                smoothed[:automatic_margin_trimming_buffer] = 0
                smoothed[-automatic_margin_trimming_buffer:] = 0
            else:
                automatic_margin_trimming_buffer = 0

            #The expression "np.where(condition)" is shorthand for "condition.nonzero()". 
            #This function returns one array of indices for each dimension of the input.
            #For a 1D array (our case), it returns a tuple with 1 element "(array([...]),)",
            #hence the need to index the array at the index zero.
            indices = np.where(smoothed)[0]
            #The conditions "indices.size > 0" and "cropped_width > horizontal_crop_kernel_threshold"
            #will ensure that a block of text (and not a speck of dust or a splatter of ink) was detected.
            if indices.size > 1:
                #The width of the cropped image with the extra horizontal space trimmed is
                #claculated by subtracting the first index of the array "indices" from the
                #last index of that array of contiguous pixels detected in the convolution step.
                cropped_width = indices[-1] - indices[0] 

                if cropped_width > horizontal_crop_kernel_threshold:
                    horizontal_cropping_successful = True
                    left_cropping_index = indices[0]
                    right_cropping_index = indices[-1]

                    #We only add extra padding if the pages were much narrower than
                    #they should ("cropped_width < 2/3 * width"), and the text started/ended
                    #very close to one of the vertical edges of the scanned pages. This will
                    #properly center the page.
                    extra_padding_left = 0
                    extra_padding_right = 0
                    narrow_page_threshold = 2/3 * width
                    #If the cropped width is very narrow (below the threshold "narrow_page_threshold"),
                    #then the cropped image will be "widened" (it won't be cropped as much) by an amount
                    #equal to "padding_width", which is calculated by halving the difference between either 
                    #the average width of the pages of the book (if the length of "list_of_cropped_page_widths"
                    #is greater than zero), or the "narrow_page_threshold" and the cropped width, effectively 
                    #bringing the width of that cropped page to the average cropped width in the former case, 
                    #and to the "narrow_page_threshold" in the latter case.                
                    if cropped_width < narrow_page_threshold:
                        padding_width = round((narrow_page_threshold - cropped_width)/2)

                        #The function "get_list_average_value()" will return the
                        #average value of a list of digits, provided that the list
                        #isn't empty, in which case it will return "None".
                        average_cropped_page_width = get_list_average_value(list_of_cropped_page_widths)

                        if (average_cropped_page_width != None and
                            average_cropped_page_width > narrow_page_threshold):
                                padding_width = round((average_cropped_page_width - cropped_width)/2)

                        #If there are at least "padding_width" pixels to the left of the
                        #left cropping point ("indices[0]"), then the left margin will
                        #be brought back by "padding_width" pixels.
                        if indices[0] - padding_width > 0:
                            left_margin = indices[0] - padding_width
                        #Otherwise, the image will start from its left 
                        #cropping point ("indices[0]"), but be padded 
                        #to the left with "padding_width" pixels
                        #("extra_padding_left = padding_width").
                        else:
                            left_margin = indices[0]
                            extra_padding_left = padding_width
                        #If there are less than "padding_width" pixels to the
                        #right of the right cropping point ("indices[-1]"), 
                        #then the right margin will end at the right cropping
                        #point ("indices[-1]"), but be padded  to the right
                        #with "padding_width" pixels ("extra_padding_right = padding_width").  
                        if indices[-1] + padding_width > width:
                            right_margin = indices[-1]
                            extra_padding_right = padding_width
                        #Otherwise the right margin will be brought back
                        #by "padding_width" from the right cropping point
                        #("indices[-1]").
                        else:
                            right_margin = indices[-1] + padding_width
                    #If the page is at least as wide as the "narrow_page_threshold",
                    #then the "else" statement below will run.
                    else:
                        #When cropping the page horizontally, a margin trimming 
                        #buffer will expand the crop selection by a number of 
                        #pixels proportional to the original page, to ensure 
                        #that no text is lost, and to preserve some kind of 
                        #margin for the block of text.

                        #If the left cropping point ("indices[0]") is greater than
                        #the value of "automatic_margin_trimming_buffer" (meaning that
                        #there are at least "automatic_margin_trimming_buffer" pixels
                        #between the zero "x" coordinate and the left cropping point ("indices[0]")), 
                        #then the left cropping point will be brought back by that value from
                        #the initial left cropping point to avoid cropping out some text.
                        if indices[0] > automatic_margin_trimming_buffer:
                            left_margin = indices[0] - automatic_margin_trimming_buffer
                        #Otherwise, the left margin will be set to zero and no cropping 
                        #will take place, so as to avoid cropping any text.
                        else:
                            left_margin = 0
                        #If the right cropping point ("indices[-1]") is less than 
                        #"automatic_margin_trimming_buffer" pixels from the right
                        #edge of the original page ("width"), then no cropping will
                        #take place on the right edge of the page to avoid cropping 
                        #out some text ("right_margin = width")
                        if indices[-1] + automatic_margin_trimming_buffer > width:
                            right_margin = width
                        #Otherwise, the right cropping point will be pushed further
                        #to the right by "automatic_margin_trimming_buffer" pixels,
                        #so as to avoid cropping out some text.
                        else:
                            right_margin = indices[-1] + automatic_margin_trimming_buffer
            #The potentially blank page index (+1 as it is zero-indexed)
            #is added to the set "set_of_potential_blank_pages", as the
            #the list of convolution chunk pixel indices ("indices") does 
            #not contain more than one element, meaning it is impossible 
            #to have the two edges to the block of text that are required
            #for cropping the page.
            else:
                set_of_potential_blank_pages.add(page_index + 1)
        #The potentially blank page index (+1 as it is zero-indexed)
        #is added to the set "set_of_potential_blank_pages", as the
        #page's maximum sum of pixels in the non-white pixel histogram 
        #("maximum_sum") doesn't pass the minimum non-white pixel 
        #threshold to be considered a page with text content.
        else:
            set_of_potential_blank_pages.add(page_index + 1)
        #The columns are added for each row of "img_array_cropping",
        #(hence the "axis = 1"), in order to tally up the
        #number of black pixels (1s) in each row. This 
        #will allow to check for contiguity when determining
        #the vertical margins of the text block.     
        line_sums = np.sum(img_array_cropping, axis=1)
        #The maximum value of "line_sums"1D horizontal array
        #will allow to determine if the page is likely a blank page
        #("maximum_sum <= width * 0.02") or a page that
        #contains content otherwise ("if" statement below).
        maximum_sum = np.max(line_sums)
        #If contiguous non-white pixels within the threshold of the kernel radius were detected horizontally and vertically
        #("horizontal_cropping_successful == True and vertical_cropping_successful == True"), then the NumPy array will be cropped.
        vertical_cropping_successful = False
        #The pixel detection threshold should be according to the page width,
        #as all of the non-white column pixels are added together.
        if maximum_sum > width * 0.02:
            #A minimum threshold of black pixels added along all columns for each row ("line_sums")
            #of one percent of the initial page width +5 is used as a threshold to say that there
            #is content in that given row. 

            #+5 added in case the value of np.max(line_sums) is very small, as in a blank page.
            #This ensures that absolute silence (0 pixels) never triggers as the "content"

            #The "flatten()" method will collapse the array to a 1D array, as the kernel is also 1D
            #and will check for contiguity across each row of black pixels (within the bounds of 
            #"vertical_crop_kernel_threshold").
            has_content = (line_sums > 0.01 * width + 5).astype(np.uint8).flatten()

            #A NumPy convolution operation will be performed in order to detect
            #contiguous vertical pixels belonging to the block of text. The kernel
            #is comprised of a 1D NumPy array (1D as we are traversing a 1D array of 
            #black and white pixels) initialized with ones, and its values will be 
            #incremented each times it crosses black pixels (1s) in the inverted page 
            #NumPy array. Larger kernel sizes will be able to "see" pixels across larger 
            #gaps, but might result in the inclusion of more noise. It is a balancing act, 
            #but generally, a kernel size of about 2% of the initial page height should 
            #give reasonable results in detecting contiguous rows of black pixels,
            #resulting from the addition of all black pixels along all columns for these
            #given rows, within the range of acceptable white pixel gaps as per the 
            #kernel size and kernel radius. 

            #The vertical_kernal_size is also proportional to the initial page height, as the
            #page height was also used to determine the kernel size on the vertical axis and 
            #we are detecting the same characters, but in the other dimension (it will keep
            #things more consistent).
            vertical_crop_kernel_size = round(vertical_crop_kernel_size_height_percent * height)
            #The kernel radius, expressed as a percentage of the kernel size, determines 
            #what overlap of black pixels within the kernel is required in order for them to be 
            #blurred together in the convolution step. The maximum value for this is the kernel 
            #size itself (complete overlap, or 100% kernel size), which wouldn't allow for any 
            #white pixels (gaps). A value of around 30% the kernel size is usually good for 
            #detecting contiguous columns of black pixels (detecting the left and right edges 
            #of the text), while around 20% of the kernel size is used when detecting contiguous 
            #rows of black pixels (detecting the top and bottom edges of the text). A smaller 
            #threshold is used in the latter case because there can be empty lines in-between
            #paragraphs, or larger vertical spaces between the end of a chapter and the beginning
            #of the next chapter. 

            #If you need to cover greater vertical gaps when detecting the top and bottom edges 
            #of the page, you will likely need to both increase the kernel size from its initial 
            #value of 8% of the initial height of the page, and potentially decrease the kernel 
            #threshold from its value of 20% of the adjusted kernel size. 
            vertical_crop_kernel_threshold = round(vertical_crop_kernel_radius_kernel_size_percent * vertical_crop_kernel_size)

            kernel = np.ones(vertical_crop_kernel_size, dtype=np.uint8)
            #The parameter "mode='same'" will ensure that the output array from the convolution
            #step is the exact same length as the input image height. This will allow to map where
            #the top and bottom edges of the image are in the original image, based on the output layer's
            #first and last indices that meet the threshold requirement.

            try:
                #The "vertical_crop_kernel_threshold" is the minimum number of "hits" in a kernel window 
                #("vertical_crop_kernel_radius_kernel_size_percent" times the kernel size) to call it a block of text.
                smoothed = np.convolve(has_content, kernel, mode='same') >= vertical_crop_kernel_threshold
            except Exception as e:
                #The function "get_terminal_dimensions()" will return the number of columns 
                #and rows in the console, to allow to properly format the text and dividers.
                columns, lines = get_terminal_dimensions()
                error_string = textwrap.fill("Please either increase the value of 'Top-Bottom Crop Kernel Size Percentage' and/or 'Top-Bottom Crop Kernel Radius Percentage', as no contiguous black pixels were detected during the vertical convolution step when cropping the top and bottom margins of the pages.", width=columns)

                print("\n" + "=" * columns)
                print("CRITICAL ERRROR ENCOUNTERED")
                print("\nDetails:", e)
                print("\n" + "=" * error_string)

                #The function "write_entry_in_error_log()" will write 
                #the full technical traceback error to the error log.
                write_entry_in_error_log()

                sys.exit(error_string + "\n")

            #0.02 times the page height pixels are set to white (zero)
            #in the "smoothed" output array in order to avoid registering 
            #the scanned image's vertical edges.
            excluded_pixels_height_when_cropping = round(0.02 * height)
            smoothed[:excluded_pixels_height_when_cropping] = 0
            smoothed[-excluded_pixels_height_when_cropping:] = 0 

            #When cropping the page vertically, a margin trimming 
            #buffer will expand the crop selection by a number of 
            #pixels proportional to the original page, to ensure 
            #that no text is lost, and to preserve some kind of 
            #margin for the block of text.

            #This means that it should be relatively safe to make all
            #pixels within these buffers white (zero) in "smoothed",
            #as they will be added back anyways, provided that some
            #text is detected right after the buffers. Doing this 
            #will help to crop out the horizontal edges of the 
            #page images.
            if vertical_crop_margin_buffer_height_percentage > 0:
                automatic_margin_trimming_buffer = round(vertical_crop_margin_buffer_height_percentage * height)
                smoothed[:automatic_margin_trimming_buffer] = 0
                smoothed[-automatic_margin_trimming_buffer:] = 0 
            else:
                automatic_margin_trimming_buffer = 0

            #The expression "np.where(condition)" is shorthand for "condition.nonzero()". 
            #This function returns one array of indices for each dimension of the input.
            #For a 1D array (our case), it returns a tuple with 1 element "(array([...]),)",
            #hence the need to index the array at the index zero.
            indices = np.where(smoothed)[0]
            #The conditions "indices.size > 0" and "cropped_height > vertical_crop_kernel_threshold"
            #will ensure that a block of text (and not a speck of dust or a splatter of ink) was detected.
            if indices.size > 1:     
                #The height of the cropped image with the extra vertical space trimmed is
                #claculated by subtracting the first index of the array "indices" from the
                #last index of that array of contiguous pixels detected in the convolution step.
                cropped_height = indices[-1] - indices[0] 

                if cropped_height > vertical_crop_kernel_threshold:
                    vertical_cropping_successful = True
                    top_cropping_index = indices[0]
                    bottom_cropping_index = indices[-1]

                    #Here, no extra space is added for pages that are very short, as
                    #pages are generally scaled according to the page widths within the
                    #PDF document (this will avoid panning across blank space when reading
                    #on an eReader).

                    #When cropping the page vertically, a margin trimming 
                    #buffer will expand the crop selection by a number of 
                    #pixels proportional to the original page, to ensure 
                    #that no text is lost, and to preserve some kind of 
                    #margin for the block of text.

                    #If the top cropping point ("indices[0]") is greater than
                    #the value of "automatic_margin_trimming_buffer" (meaning that
                    #there are at least "automatic_margin_trimming_buffer" pixels
                    #between the zero "y" coordinate and the top cropping point ("indices[0]")), 
                    #then the top cropping point will be brought back by that value from
                    #the initial top cropping point to avoid cropping out some text.
                    if indices[0] > automatic_margin_trimming_buffer:
                        top_margin = indices[0] - automatic_margin_trimming_buffer
                    #Otherwise, the top margin will be set to zero and no cropping 
                    #will take place, so as to avoid cropping any text.
                    else:
                        top_margin = 0
                    #If the bottom cropping point ("indices[-1]") is less than 
                    #"automatic_margin_trimming_buffer" pixels from the bottom
                    #edge of the original page ("width"), then no cropping will
                    #take place on the bottom edge of the page to avoid cropping 
                    #out some text ("bottom_margin = height")
                    if indices[-1] + automatic_margin_trimming_buffer > height:
                        bottom_margin = height
                    #Otherwise, the bottom cropping point will be pushed further
                    #down by "automatic_margin_trimming_buffer" pixels, so as to
                    #avoid cropping out some text.
                    else:
                        bottom_margin = indices[-1] + automatic_margin_trimming_buffer
            #The potentially blank page index (+1 as it is zero-indexed)
            #is added to the set "set_of_potential_blank_pages", as the
            #the list of convolution chunk pixel indices ("indices") does 
            #not contain more than one element, meaning it is impossible 
            #to have the two edges to the block of text that are required
            #for cropping the page.
            else:
                set_of_potential_blank_pages.add(page_index + 1)
        #The potentially blank page index (+1 as it is zero-indexed)
        #is added to the set "set_of_potential_blank_pages", as the
        #page's maximum sum of pixels in the non-white pixel histogram 
        #("maximum_sum") doesn't pass the minimum non-white pixel 
        #threshold to be considered a page with text content.
        else:
            set_of_potential_blank_pages.add(page_index + 1)

        #If contiguous non-white pixels within the threshold of the kernel radius were detected horizontally and vertically
        #("horizontal_cropping_successful == True and vertical_cropping_successful == True"), then the NumPy array will be cropped.
        if horizontal_cropping_successful and vertical_cropping_successful:

            #The center of the original grayscale image "img_array" that was lightly filtered so as to preserve the anti-aliasing pixels
            #will be "pasted" onto the heavily filtered image array before the inversion of the polarity ("img_array_cropping_before_inversion"),
            #in order to ensure that the margins are nice and clean, as both a heavy full page initial filter and a stringent margins filter
            #were applied to "img_array_cropping".

            #The central portion of the "img_array_cropping_before_inversion" sliced at the cropping coordinates will be replaced
            #with the grayscale values of the lightly filtered "img_array" to give the nice text with anti-aliasing and the clean
            #white outer margins of the heavily filtered "img_array_cropping_before_inversion".
            img_array_cropping_before_inversion[top_cropping_index:bottom_cropping_index, left_cropping_index:right_cropping_index] = (
                img_array[top_cropping_index:bottom_cropping_index, left_cropping_index:right_cropping_index]
            )

            #The image is cropped while retaining "y" pixels between "top_margin" and
            #"bottom_margin", and retaining "x" pixels between "left_margin" and "right_margin".
            #These margins differ from "top_cropping_index", "bottom_cropping_index",
            #"left_cropping_index" and "right_cropping_index" in that they extend the crop
            #area by the safe vertical and horizontal zones ("vertical_crop_margin_buffer_height_percentage * height")
            #and ("horizontal_crop_margin_buffer_width_percentage * width").
            img_array_cropping_before_inversion = img_array_cropping_before_inversion[top_margin:bottom_margin, left_margin:right_margin]

            #If the cropped width is very narrow (below the threshold "narrow_page_threshold"),
            #then the cropped image will be "widened" (it won't be cropped as much) by an amount
            #equal to "padding_width", which is calculated by halving the difference between either 
            #the average width of the pages of the book (if the length of "list_of_cropped_page_widths"
            #is greater than zero), or the "narrow_page_threshold" and the cropped width, effectively 
            #bringing the width of that cropped page to the average cropped width in the former case, 
            #and to the "narrow_page_threshold" in the latter case.
            if extra_padding_left > 0 or extra_padding_right > 0:
                np.pad(img_array_cropping_before_inversion, ((0, 0), (extra_padding_left, extra_padding_right)), 
                    mode='constant', constant_values=0)

            #The "img_array" is updated with the cropping changes.   
            img_array = img_array_cropping_before_inversion.copy()

        #If the cropped width is very narrow (below the threshold "narrow_page_threshold"),
        #then the cropped image will be "widened" (it won't be cropped as much) by an amount
        #equal to "padding_width", which is calculated by halving the difference between either 
        #the average width of the pages of the book (if the length of "list_of_cropped_page_widths"
        #is greater than zero), or the "narrow_page_threshold" and the cropped width, effectively 
        #bringing the width of that cropped page to the average cropped width in the former case, 
        #and to the "narrow_page_threshold" in the latter case.
        list_of_cropped_page_widths.append(img_array.shape[1])

    #If the pages are to be set to black and white,
    #every non-white pixel will be set to black (zero).
    if black_and_white_mode_enabled:
        img_array[img_array != 1] = 0
    #If "is_dark_mode_enabled" mode is enabled,
    #then the array will be inverted in polarity.
    #This needs to be another "if" statement, in
    #case the user has selected both the 
    #"Black and White Mode" and the "Dark Mode".
    if is_dark_mode_enabled:
        img_array = 1 - img_array

    #Convert the NumPy array back to a grayscale pixmap, after denormalizing from 0.0-1 to 0-255:
    samples_uint8 = (img_array * 255).astype(np.uint8).tobytes()

    #The shape is (height, width, 1), as it is in grayscale
    #To create a Pixmap from raw data, positional arguments must be provided 
    #(without keywords such as "colorspace", "width" or "height")
    new_pixmap = pymupdf.Pixmap(
                pymupdf.csGRAY, 
                img_array.shape[1], #Width (columns) 
                img_array.shape[0], #Height (rows)
                samples_uint8,
                False) #Indicates no alpha channel

    #The value of "cumulative_pdf_file_size_estimation" is incremented
    #with the length of the bytestream derived from "new_pixmap", as 
    #an approximation of the current size of the PDF document. This 
    #does not factor in the optimization steps that go into reducing
    #the file of the final PDF document, however. You might need to
    #specify a larger target file size to account for this.
    cumulative_pdf_file_size_estimation += len(new_pixmap.tobytes())

    #Create a new page with the same dimensions as the Pixmap object "new_pixmap"
    new_page = doc_output.new_page(width=new_pixmap.width, height=new_pixmap.height)

    #Insert the Pixmap object "new_pixmap" directly
    #"rect" defines where the image goes (new_page.rect fills the whole page)
    new_page.insert_image(new_page.rect, pixmap=new_pixmap)

    list_of_original_document_page_numbers.append(page_index + 1)

    return (doc_output,
            cumulative_pdf_file_size_estimation, 
            list_of_cropped_page_widths,
            set_of_potential_blank_pages,
            list_of_original_document_page_numbers)


#The function "display_progress()" will display the progress string in the console
#and return the estimated number of seconds for the code to complete.
def display_progress(page_index, first_page, last_page, start_time, previous_estimated_seconds, list_of_individual_removed_pages):

    elapsed_seconds = time.perf_counter() - start_time
    #divmod returns (minutes, remaining_seconds)
    mins, secs = divmod(round(elapsed_seconds), 60)
    time_string = f"{mins:02}:{secs:02}"
    eta_string = ""

    #If the final PDF will only be comprised of one page,
    #then the "percent_completion" will be 100% after 
    #processing that page (so as to avoid "Division 
    #by Zero" errors).
    if last_page - first_page == 0:
        percent_completion = 100
        eta_string = f" ETA: 00:00\n\nGenerating final PDF file (this could take a minute).\n"
    else:
        #The percent completion is calculated by dividing the difference between the current page index
        #and the first page index by the total number of pages to be processed, which is itself
        #calculated by subtracting the first page index from the last page index. The resulting quotient 
        #is multiplied by 100 and then rounded when printed on-screen.
        percent_completion = (page_index-first_page)/(last_page-first_page) * 100

    #The previous estimation of the remaining number of seconds is stored in the variable
    #"previoous_estimated_seconds" and will be used instead of the current calculation
    #if it exceeds the previous estimation, so as to avoid the ETA timer increasing 
    #its estimation.
    estimated_seconds = previous_estimated_seconds
    #A delay of 10 pages is used to be able to gather a somewhat accurate value
    #of the elapsed time for a given percent completion value.
    if (last_page > first_page + 10 and page_index > first_page + 10):
        #The estimated number of seconds left is calculated by doing the cross-multiplication between
        #the number of percentage points left to reach completion ("100 - percent_completion") 
        #and the elapsed time for the current percent completion.
        estimated_seconds = round((100 - percent_completion) * elapsed_seconds / percent_completion)
        #The previous estimation of the remaining number of seconds is stored in the variable
        #"previoous_estimated_seconds" and will be used instead of the current calculation
        #if it exceeds the previous estimation, so as to avoid the ETA timer increasing 
        #its estimation.
        if (previous_estimated_seconds != 0 and estimated_seconds > previous_estimated_seconds):
            estimated_seconds = previous_estimated_seconds

        #If the current page index is the last page index,
        #then the remaining time is zero seconds (" ETA: 00:00").
        if (page_index == last_page):
            percent_completion = 100
            eta_string = f" ETA: 00:00\n\nGenerating final PDF file (this could take a minute).\n"
        #If the current "page_index" is the last remaining page to be processed and all the remining pages after it have been removed 
        #("[index for index in range(page_index, last_page + 1) if index + 1 not in list_of_individual_removed_pages] == [page_index]"),
        #with +1 being added because "page_index" is zero indexed, while the pages in the "list_of_individual_removed_pages" are not 
        #zero indexed, then the remaining time is zero seconds (" ETA: 00:00"). 
        elif (list_of_individual_removed_pages != [] and 
        [index for index in range(page_index, last_page + 1) if index + 1 not in list_of_individual_removed_pages] == [page_index]):
            percent_completion = 100
            eta_string = f" ETA: 00:00\n\nGenerating final PDF file (this could take a minute).\n"
        #We do not want to display negative times, hence the
        #condition ("elif (estimated_seconds > 0)").
        elif (estimated_seconds > 0):
            eta_mins, eta_secs = divmod(round(estimated_seconds), 60)
            eta_string = f" ETA: {eta_mins:02}:{eta_secs:02}"

    #"\r" resets the line
    sys.stdout.write(f"\rCompleted pages: {page_index+1} of {last_page+1} ({round(percent_completion)}%) Time: {time_string}{eta_string}")

    return estimated_seconds


#The function "get_pdf_file_path()" will assemble the current PDF file path from the current
#working folder ("cwd"), the output folder name, the file name (which may need to be truncated 
#in order for the path to be under 260 characters) and the output file number.
def get_pdf_file_path(cwd, output_folder_name, file_name, output_pdf_file_number):

    pdf_file_path = os.path.join(cwd, output_folder_name, f"{file_name} (Part {output_pdf_file_number}).pdf")

    #The file name will be truncated until 
    #the file path is 255 characters or less.
    truncated_file_name = file_name
    while (len(pdf_file_path) > 255):
        truncated_file_name = truncated_file_name[:-1]
        pdf_file_path = os.path.join(cwd, output_folder_name, f"{truncated_file_name} (Part {output_pdf_file_number}).pdf")

    return pdf_file_path

#The function "get_terminal_dimensions()" will return the number of columns 
#and rows in the console, to allow to properly format the text and dividers.
def get_terminal_dimensions():
    #Detect columns (width) and lines (height)
    #Returns a named tuple; default fallback is (80, 24)
    size = shutil.get_terminal_size(fallback=(80, 24))
    return int(size.columns * 0.75), int(size.lines)

#The function "is_valid_positive_non_zero_int" will validate the data stored in 
#the dictionary obtained from the "json_settings.json" file to make sure it is
#not "NaN" or "Infinity" (not "math.isfinite(number)") and make sure that the 
#"int" version of the number is equal to itself to exclude non-round floating 
#point numbers (ex: 1.2 instead of 1.0) and also exclude negative and zero 
#numbers "number <= 0". It will return "True" if the number is a valid
#("if" statement) integer and "False" otherwise ("else" statement).
def is_valid_positive_non_zero_int(number):
    if not math.isfinite(number) or number != math.floor(number) or number <= 0: 
        return False
    else:
        return True

#The function "is_valid_non_negative_int_or_float" will validate the data stored 
#in the dictionary obtained from the "json_settings.json" file to make sure 
#it is not "NaN" or "Infinity" (not "math.isfinite(number)") and make sure 
#that the number is either an integer or a float and also exclude negative 
#numbers "number < 0". It will return "True" if the number is a valid
#("if" statement) integer and "False" otherwise ("else" statement).   
def is_valid_non_negative_int_or_float(number):
    if not math.isfinite(number) or not isinstance(number, (int, float)) or number < 0:
        return False
    else:
        return True

#The function "is_valid_int_or_float" will validate the data stored 
#in the dictionary obtained from the "json_settings.json" file to make sure 
#it is not "NaN" or "Infinity" (not "math.isfinite(number)") and make sure 
#that the number is either an integer or a float. It will return "True" if 
#the number is a valid ("if" statement) integer and "False" otherwise 
#("else" statement).   
def is_valid_int_or_float(number):
    if not math.isfinite(number) or not isinstance(number, (int, float)):
        return False
    else:
        return True


#The function "return_on_for_true_and_off_for_false()" will
#return "ON" if the vlaue of the Boolean argument was "True"
#and "OFF" otherwise.
def return_on_for_true_and_off_for_false(boolean):
    if boolean:
        return "ON"
    else:
        return "OFF"


#The function "atomic_save()" will create a temporary JSON file with the updated changes.
#If the files is created successfully, then the files will be swapped. If a problem is 
#encountered, the temp file will be unlinked and an error log will be reported.
def atomic_save(json_settings_dictionary, json_settings_file_path_name):
    #Create a temp file in the same directory
    temp_dir = os.path.dirname(json_settings_file_path_name) or "."
    json_file_descriptor, temp_path = tempfile.mkstemp(dir=temp_dir, text=True)

    try:
        with os.fdopen(json_file_descriptor, "w", encoding="utf-8") as f:
            #Write the default values found in "json_settings_dictionary" in the empty JSON file, 
            #with four space indentations to make it more human-readable.
            json.dump(json_settings_dictionary, f, indent=4)
            #Ensure the data is flushed to hardware.
            f.flush()
            #"os.fsync(f.fileno())" is required to force the OS to physically commit
            #every bit of information to the hardware storage right now, preventing 
            #a situation where an empty file might be created if the computer crashed
            #before the OS finished waiting before committing the file to memory. 
            os.fsync(f.fileno())

        #Swap the files only if the temp file was successfully generated (Atomic security)
        os.replace(temp_path, json_settings_file_path_name)
    except Exception as e:
        #Clean up temp file if something goes wrong BEFORE the swap
        if os.path.exists(temp_path):
            os.unlink(temp_path)

        #The function "write_entry_in_error_log()" will write 
        #the full technical traceback error to the error log.
        write_entry_in_error_log()

        #The function "get_terminal_dimensions()" will return the number of columns 
        #and rows in the console, to allow to properly format the text and dividers.
        columns, lines = get_terminal_dimensions()

        print("\n" + "=" * columns)
        print("CRITICAL ERRROR ENCOUNTERED")
        print("\nDetails:", e)
        print("\n" + "=" * columns)

        #Exit with error code
        sys.exit(1)


#The function "get_last_page_string()" will return "Last Page of Original PDF"
#if the current "Last Page" setting is set to zero, and the string version of
#"json_settings_dictionary["Last Page"]" otherwise.
def get_last_page_string(json_settings_dictionary):        
    #If the current value of the JSON key "Last Page" is zero, meaning
    #continue until the end of the original PDF document, the string
    #"Last Page of Original PDF" will be stored in "last_page_string".
    if json_settings_dictionary["Last Page"] == 0:
        last_page_string = "Last Page of Original PDF"
    #Otherwise the string version of the value of the JSON key "Last Page"
    #will be stored in "last_page_string".
    else:
        last_page_string = str(json_settings_dictionary["Last Page"])
    return last_page_string


#The "get_cover_page_font_size()" function will calculate the maximum font size for which the 
#longest of the lines of the title or author text may be displayed within the threshold of 
#80% of the cover page width. It will return the final font size that corresponds to 90% of 
#the size of the maximal font size, in order to prevent the textbox from automatically 
#wrapping the text. The user then needs to specify the locations in the file name where 
#carriage returns need to be placed by sequences of two consecutive spaces.
def get_cover_page_font_size(cover_page, cover_page_font_object, cover_page_string_list, cover_page_width_points, cover_page_height_points, cover_page_line_spacing, cover_title_font_size = None):

    #A horizontal threshold of 80% of the cover page height in points
    #is selected as the maximum width for the title or author name 
    #longest line.
    text_width_target = round(0.8 * cover_page_width_points)

    #If a cover title font size has been passed in as an optional
    #argument, then the text color will be set to white, as we are
    #now processing the author name, which will be drawn in white
    #over the black bottom half of the page. The "y0" coordinate
    #of "text_rect" is therefore 50% of the page height, and the
    #"y1" corresponds to the bottom of the page ("cover_page_height_points") 
    if cover_title_font_size: 
        text_rect = pymupdf.Rect(round(0.1 * cover_page_width_points), 
                                round(0.5 * cover_page_height_points), 
                                round(0.9 * cover_page_width_points), 
                                round(cover_page_height_points))
    #Otherwise the title text will be drawn in black color
    #in the top white half of the page.
    else:
        text_rect = pymupdf.Rect(round(0.1 * cover_page_width_points), 
                    round(0.0 * cover_page_height_points), 
                    round(0.9 * cover_page_width_points), 
                    round(0.5 * cover_page_height_points))

    #The initial font size is set to a tenth of the page height
    #in points, and will be decremented with every iteration of 
    #the "while" loop below, until the text either fits horizontally,
    #or the font size reaches one. The proportional font size starting
    #point ensures that the cover page will look similar even if the
    #starting image sizes from which the "Document" object pages are
    #derived differ from those from the original PDF documents 
    #the app was designed on.
    loop_font_size = cover_page_height_points//10
    while True: 
        #If the cover page string list is comprised of more than one line,
        #then each line will be submitted to the "text_length()" PyMuPDF method
        #and the longest length will be stored in "max_length". 
        if len(cover_page_string_list) > 1:
            #The widest line width will be used when evaluating whether or not the "while"
            #loop should be broken out of. If its length is less than the horizontal threshold 
            #of 80% of the page width, or the font size reaches one, the "while" loop will break. 
            max_length = max([cover_page_font_object.text_length(line, fontsize=loop_font_size) for line in cover_page_string_list])
        #If the cover string list is only comprised of one line,
        #then the only line at index zero of the list will be 
        #submitted to the "text_length()" PyMuPDF method to
        #and the result will be stored in "max_length".
        else:
            max_length = cover_page_font_object.text_length(cover_page_string_list[0], fontsize=loop_font_size)
        if max_length <= text_width_target or loop_font_size == 1:
            #If cover the title string font size ("cover_title_font_size") has 
            #been provided as an optional argument author string is now being 
            #processed, then an additional condition for breaking out of the
            #"while" loop is that the author text's font size must be no
            #greater than 85% of the cover title string's font size.
            if cover_title_font_size and loop_font_size > 0.85 * cover_title_font_size:
                loop_font_size -= 1
                continue
            #A "TextWriter" object ("text_wrap") is used to draft up the textbox
            #without actually drawing it on the page, in order to evaluate the 
            #return value of the "fill_textbox()" method returns an empty list,
            #which indicates that the text fit nicely within "text_rect" without
            #any overspills, in which case the function would return.
            text_wrap = pymupdf.TextWriter(cover_page.rect)

            #If the cover page string list is comprised of more than one line,
            #it will be joined into one string with carriage return characters
            #("\n") and stored into "textbox_string". 
            if len(cover_page_string_list) > 1:
                textbox_string = ("\n").join(cover_page_string_list)
            #If the cover string list is only comprised of one line,
            #then the string at index zero of the list will be used
            #as the "textbox_string."
            else:
                textbox_string = cover_page_string_list[0]

            fill_textbox_result = text_wrap.fill_textbox(
                text_rect,
                textbox_string,
                font=cover_page_font_object,
                fontsize = loop_font_size,
                lineheight=cover_page_line_spacing,
                align=pymupdf.TEXT_ALIGN_CENTER
            )
            if (fill_textbox_result == []):
                return round(0.9 * loop_font_size)
        #If the conditions are not met, then
        #the font size will be decremented for
        #the next "while" loop iteration.
        loop_font_size -= 1


#The "split_title_author_string_for_carriage_returns()"
#function will split a title page string along sequences 
#of two or more successive spaces denoting the location 
#of carriage returns. It will allow for one or more
#successive carriage returns (four spaces for two
#carriage returns, for example). It returns the
#split string, with the "\n" characters added to their
#appropriate places and the extra spaces trimmed out.
def split_title_author_string_for_carriage_returns(cover_string):
    #The list "new_split_cover_string" will only contain
    #the lines of text, with the added carriage returns
    #instead of the sequences of two or more consecutive
    #spaces.
    new_split_cover_string = []
    #The original cover page string is split while retaining sequences
    #of two or more successive spaces as distinct elements in the list.
    split_cover_string = re.split(r"([ ]{2,})", cover_string.strip())
    #The elements of the "split_cover_string" are cycled over,
    #and if the element, when stripped of space characters gives
    #an empty string, then we know that this is a marker for one
    #or more carriage returns. The number of carriage returns is
    #calculated by dividing the number of consecutive spaces by
    #two as a floor division (so that 3 spaces gives 1 carriage 
    #returns). If the new list "new_split_cover_string" contains
    #at least one line of text, then it will be appended with a 
    #new line containing as many "\n" characters as there are 
    #sequences of two consecutive space characters in excess of
    #two ("number_of_carriage_returns - 2"), or a new line with 
    #only a space if there were four consecutive spaces 
    #(number_of_carriage_returns == 2). This is done so as to
    #to avoid having "\n" characters at the end of the lines, which
    #skews the centering of the cover page preview in the CLI).
    for i in range(len(split_cover_string)):
        is_element_only_spaces = split_cover_string[i].strip(" ") == ""
        if is_element_only_spaces and len(new_split_cover_string) > 0:
            number_of_carriage_returns = len(split_cover_string[i])//2
            #For example: \n\n\n at the end of an element of the list is equivalent 
            #to an extra element in the list with only "\n" (as it stands on its 
            #one line and the lines before it and after it are already split)
            if number_of_carriage_returns > 2:
                new_split_cover_string.append("\n" * (number_of_carriage_returns - 2))
            if number_of_carriage_returns == 2:
                new_split_cover_string.append(" ")
        #If the element is not only comprised of space characters,
        #then it will be appended to the list "new_split_cover_string".
        elif not is_element_only_spaces:
            new_split_cover_string.append(split_cover_string[i])
    return new_split_cover_string


#The function "get_cover_page_color_string()" will retrieve the color string corresponding to the
#tuple of the chosen color in "colors_dict". If the color tuple isn't in "colors_dict", then it
#means that the user has provided a custom color, so its RGB and Hex code information will be
#returned in string form instead.
def get_cover_page_color_string(json_settings_dictionary):
    #Get the color string value at the key of the tuple form of the RGB information 
    #of the color, and the list of RGB values for the custom color otherwise. 
    color_string = colors_dict.get(tuple(json_settings_dictionary["Cover Page Color"]), list(json_settings_dictionary["Cover Page Color"]))
    #If the custom color list was returned by the "get()" method,
    # the custom color's RGB and Hex code information will be
    #returned in string form.
    if isinstance(color_string, list):
        color_string = f"Custom Color: rgb({color_string[0]}, {color_string[1]}, {color_string[2]}) | Hex: #{color_string[0]:02x}{color_string[1]:02x}{color_string[2]:02x}"
    return color_string


#The "save_pdf()" function will generate a cover page (if the "Cover Page" mode is enabled)
#and output the PyMuPDF "Document" object as a PDF file with the corresponding output PDF
#file number in parentheses (e.g.,: "Book Title - Subtitle by Author Name (Part 1).pdf").
def save_pdf(cwd,
    doc_output,
    list_of_original_document_page_numbers,
    is_dark_mode_enabled,
    do_crop_pages,
    do_pad_pages,
    cover_page_enabled,
    cover_page_color,
    cover_page_line_spacing,
    split_title_string_list,
    split_author_string_list,
    cumulative_pdf_file_size_estimation,
    output_pdf_file_number,
    output_folder_name,
    output_file_name,
    set_of_potential_blank_pages):

    #If the user has selected white as their white color
    #cover_page_color == (1.0,), then it means that
    #the cover page will be generated in black and white
    #colors only, so the grayscale values may be used
    #(one-member tuples), whereas three-member RGB
    #tuples will be used if the light color is not
    #white.
    if cover_page_color == (1, 1, 1):
        cover_page_color = (1,)
        cover_page_black_color = (0,)
    else:
        cover_page_black_color = (0, 0, 0)


    doc_output_page_heights_list = []
    doc_output_page_widths_list = []

    #The lists of the "doc_output" page heights and widths are
    #populated with the current page dimensions. 
    for page in doc_output:
        doc_output_page_heights_list.append(page.rect.height)
        doc_output_page_widths_list.append(page.rect.width)
    #The average page height and width within "doc_output" are
    #calculated and will be used to calculate the average aspect 
    #ratio of the pages, which will allow to locate potential
    #blank pages, which are typically cropped horizontally
    #(left-right) very extensively, and consequently have
    #a width/height aspect ratio that is much lower than
    #regular pages of text.
    doc_output_average_height = sum(doc_output_page_heights_list)/len(doc_output_page_heights_list)
    doc_output_average_width = sum(doc_output_page_widths_list)/len(doc_output_page_widths_list)

    doc_output_average_width_over_height = doc_output_average_width / doc_output_average_height
    #A threshold of two-thirds of the average width/height ratio is
    #used to designate pages that are potentially blank pages.
    narrow_page_threshold = 2/3 * doc_output_average_width_over_height
    #A list of page numbers that have an aspect ratio ("doc_output[i].rect.width/doc_output[i].rect.height")
    #inferior to the "narrow_page_threshold" are tallied in the "list_of_narrow_pages", and will be added to
    #the "set_of_potential_blank_pages" that included pages that were not cropped because of a low level of
    #detected non-white pixels. 

    #The "list_of_original_document_page_numbers" will tally up the list of
    #all the page numbers in the original PDF document that were included in
    #"doc_output". This will allow to determine which cropped pages were 
    #horizontally cropped much more than other pages and are thus likely
    #blank pages. As the page numbers in "list_of_original_document_page_numbers"
    #line up with the actual pages of "doc_output", both of these can be indexed
    #with the same index "i".
    list_of_narrow_pages = [list_of_original_document_page_numbers[i] for i in range(len(doc_output)) if doc_output[i].rect.width/doc_output[i].rect.height < narrow_page_threshold]
    if len(list_of_narrow_pages) > 0:
        set_of_potential_blank_pages = set_of_potential_blank_pages.union(set(list_of_narrow_pages))
    #If the "Auto-Padding" mode is enabled, then the largest page width
    #and height of all the pages will be used as the final document 
    #page dimensions, which will allow to individualize padding on
    #a page by page basis.
    if do_crop_pages and do_pad_pages:
        #The padded page dimensions will correspond to the maximal page
        #height and width found in all the pages of "doc_output".
        output_page_height = max(doc_output_page_heights_list)
        output_page_width = max(doc_output_page_widths_list)

        #The output "Document" object "doc_output" will by cycled over
        #in reversed order in order to avoid indexing issues while inserting
        #and deleting pages.
        for i in range(len(doc_output)-1, -1, -1):
            #The extra padding pixels on either side of the page is calculated by subtracting
            #the original page dimension from the final page dimension, and halving that result.
            extra_horizontal_padding = round((output_page_width - doc_output[i].rect.width)/2)
            extra_vertical_padding = round((output_page_height - doc_output[i].rect.height)/2)
            #A new page is created in the document at the index "i" of the current page, which
            #shifts what was originally page "i" to "i+1", with the new page now at index "i".
            #The new page is generated with the final PDF page dimensions based on the maximal
            #page width and height in the entire set of cropped pages.
            new_page = doc_output.new_page(i, width=output_page_width, height=output_page_height) 

            #If the "Dark Mode" is enabled, the background of
            #the padded pages will be set to black ("(0, 0, 0)")
            #by drawing a rectangle the same size as the "new_page"
            #rectangle, with no border ("color=None").
            if is_dark_mode_enabled:
                new_page.draw_rect(new_page.rect, color=None, fill=(0,0,0), overlay=False)

            #A temporary "Document" object is created in order to insert the page that 
            #we wish to pad, as we cannot copy and paste pages within the same document.
            temp_doc = pymupdf.open()
            #The original page (which is now at the index "i+1") is inserted into
            #the temporary "Document" object "temp_doc". It is important to avoid
            #deleting page "i+1" in "doc_output" until we have successfully performed
            #the "show_pdf_page()" operation to stamp the page from the temporary
            #document onto our page "i" to avoid errors.
            temp_doc.insert_pdf(doc_output, from_page=i+1, to_page=i+1)
            #The "new_rect" contains the padding information that centers the 
            #original page within the new page of dimensions "output_page_height" x
            #"output_page_width".
            new_rect = pymupdf.Rect(extra_horizontal_padding, 
                                    extra_vertical_padding,
                                    output_page_width - extra_horizontal_padding,
                                    output_page_height - extra_vertical_padding
                                    )
            #Place original content that is now at the page zero of the 
            #temporary document in your new page "i", with the padding 
            #information of "new_rect".
            new_page.show_pdf_page(new_rect, temp_doc, 0)
            #Close the temporary document before deleting
            #page "i+i" (the original page).
            temp_doc.close()
            #Delete what was formerly page "i", which was shifted
            #to "i+1" after creating a new page.
            doc_output.delete_page(i+1)

    #If both the "Auto-Cropping" and "Cover Page" modes are enabled,
    #then the size of the cover page will be the same as all of the
    #padded pages ("output_page_width" x "output_page_height").
    if do_crop_pages and do_pad_pages and cover_page_enabled:   
        cover_page_height_points = output_page_height
        cover_page_width_points = output_page_width
    #If only the "Cover Page" mode is enabled, then the code
    #needs to find the average page width and height and use
    #these average dimensions to size the cover page.
    elif cover_page_enabled:
        doc_output_page_heights_list = []
        doc_output_page_widths_list = []
        for page in doc_output:
            doc_output_page_heights_list.append(page.rect.height)
            doc_output_page_widths_list.append(page.rect.width)
        if len(doc_output_page_heights_list) > 0:
            cover_page_height_points = round(sum(doc_output_page_heights_list)/len(doc_output_page_heights_list))
        if len(doc_output_page_widths_list) > 0:
            cover_page_width_points = round(sum(doc_output_page_widths_list)/len(doc_output_page_widths_list))
    #The following "if" statement will run if the "Cover Page"
    #mode is enabled, and will generate the cover page itself.
    if cover_page_enabled:
        half_cover_page_height = round(0.5 * cover_page_height_points)

        #A new PyMuPDF "Document" object "doc_cover_page" is created
        #to store the information of the cover page PDF document.
        #This will allow the cover page to be in color, while the
        #actual pages of the book will remain in "Grayscale" or
        #"Black and White" format, thus optimizing the file size
        #of the final PDF file.
        doc_cover_page = pymupdf.open()

        #A new page is created with the width and height of the cover page.
        doc_cover_page.new_page(width=cover_page_width_points, height=cover_page_height_points)

        #The cover page of the PDF document at page index zero
        #is stored in the "cover page" variable.
        cover_page = doc_cover_page[0]

        #Draw a black ("fill=cover_page_color") rectangle with no border ("color=none"),
        #full opacity ("opacity=1.0"), in the back of the page contents ("overlay=False")
        #The rectangle will fill the entire page ("cover_page.rect").
        cover_page.draw_rect(cover_page.rect, color=None, fill=cover_page_color, overlay=False) 

        #Draw a black ("fill=cover_page_black_color") rectangle with no border ("color=none"),
        #full opacity ("opacity=1.0"), in front of the page contents ("overlay=True")
        #The rectangle will fill the upper half of the page, where the coordinates are
        #x0, y0, x1, y1, with x0, y0 being the top-left corner and x1, y1 being the
        #bottom-right corner of the rectangle.
        cover_page.draw_rect(pymupdf.Rect(0, round(0.5*cover_page_height_points), cover_page_width_points, cover_page_height_points), 
                                color=None, fill=cover_page_black_color, overlay=True)

        #The "glob" module is used to tally a list of all
        #the OTF and TTF font files present in the application's
        #root folder.
        ttf_path = os.path.join(cwd, "*.ttf")
        ttf_files = glob.glob(ttf_path)
        otf_path = os.path.join(cwd, "*.otf")
        otf_files = glob.glob(otf_path)
        #If there is at least one TTF file in the
        #root folder, the first file in the list
        #at index zero will be used to instantiate
        #the PyMuPDF "Font" object.
        if len(ttf_files) > 0:
            cover_page_font_object = pymupdf.Font(fontfile=ttf_files[0])
            cover_page_font_name = "custom_font"
        #Similar to the "if" statement for OTF fonts.
        elif len(otf_files) > 0:
            cover_page_font_object = pymupdf.Font(fontfile=otf_files[0])
            cover_page_font_name = "custom_font"
        #The "Times Bold" MuPyPDF font will be used
        #if no TTF or OTF fonts were included in the 
        #root folder.
        else:
            #The four letter abbreviation for the default font if no
            #OTF nor TTF files were included in the root folder is
            #"tibo" for "Times Bold".
            cover_page_font_object = pymupdf.Font("tibo")
            cover_page_font_name = "Times-Bold"
        #The same font buffer is registered from rendering on the page, 
        #thus ensuring that the exact same font/font size will be used
        #when measuring and drawing.

        #The font file will automatically be embedded into the PDF upon using
        #the "insert_font()" MuPyPDF method.
        cover_page.insert_font(fontname=cover_page_font_name, fontbuffer=cover_page_font_object.buffer)

        #The "get_cover_page_font_size()" function will calculate the maximum font size for which the 
        #longest of the lines of the title or author text may be displayed within the threshold of 
        #80% of the cover page width. It will return the final font size that corresponds to 90% of 
        #the size of the maximal font size, in order to prevent the textbox from automatically 
        #wrapping the text. The user then needs to specify the locations in the file name where 
        #carriage returns need to be placed by sequences of two consecutive spaces.
        cover_page_font_size = get_cover_page_font_size(cover_page, cover_page_font_object, split_title_string_list, cover_page_width_points, cover_page_height_points, cover_page_line_spacing)

        #The total height of all the lines of the cover page title string is 
        #calculated by multiplying the difference between the font object's
        #ascenders and descenders to give the font height at a font size of
        #one, times the cover page font size, times the number of lines
        #in the string, which is calculated by adding the length of the
        #"split_title_string_list" list to the number of "\n"s in all of
        #the strings in the list.
        number_of_lines = len(split_title_string_list) + sum(line.count("\n") for line in split_title_string_list)
        line_height_without_line_spacing = (cover_page_font_object.ascender - cover_page_font_object.descender) * cover_page_font_size
        if number_of_lines > 1:
            cover_title_total_height = cover_page_line_spacing * line_height_without_line_spacing * (number_of_lines - 1) + line_height_without_line_spacing
        #If there is only one line in the title, when no line spacing needs to be
        #factored in, so the height of the title string will be equal to 
        #"line_height_without_line_spacing".
        else:
            cover_title_total_height = line_height_without_line_spacing
        #A vertical spacer will be added between the bottom of the 
        #last line of the title string and the black/white interface
        #in the middle of the page, so as to avoid having the text
        #too flush with it. The spacer is set to 10% of the page height. 
        cover_page_vertical_spacer = 0.10 * cover_title_total_height

        #A "TextWriter" object ("new_text_wrap") is used to draft up the textbox
        #without actually drawing it on the page, in order to evaluate the 
        #return value of the "fill_textbox()" method returns an empty list,
        #which indicates that the text fit nicely within "text_rect" without
        #any overspills, in which case the function would return.
        new_text_wrap = pymupdf.TextWriter(cover_page.rect, color=cover_page_black_color)

        #As the text is by default drawn at the top of the "initial_text_rect",
        #and we want it to be shifted down towards the black/white interface,
        #the top "y" coordinate of the first line of the title will be shifted
        #down by an amount of pixels equal to the difference between the
        #upper half of the page ("half_cover_page_height") and the total
        #height of the cover page title string ("cover_title_total_height"),
        #minus the vertical spacer ("cover_pge_vertical_spacer") to bring the text
        #back up (avoiding it to bee too flush against the black/white interface).
        #We then need to add the font ascender plus the font size itself in order
        #to bring the "top_y" coordinate to the baseline of the text, which is
        #what the "pos" argument in the "append()" method of the PyMuPDF 
        #"TextWrap" class uses to specify the "y" coordinate at which the
        #first character of the text will be written. The font ascender
        #plus the font size is calculated adding the descender times the
        #font size (which gives the height of the descender for the current
        #font size, and is negative) to the total line height with a line 
        #spacing of one "line_height_without_line_spacing", which was 
        #calculated as follows: "(font.ascender - font.descender) * font size".
        top_y = half_cover_page_height - cover_title_total_height - cover_page_vertical_spacer + (line_height_without_line_spacing + cover_page_font_size * cover_page_font_object.descender)

        len_split_title_string_list = len(split_title_string_list)
        #Each line in the "split_title_string_list" will be appended to the 
        #initially empty "TextWrap" object "new_text_wrap", with the top-left
        #coordinates specified by the "pos" argument.
        for i in range(len_split_title_string_list):
            #The lines are horizontally centered by halving the difference between the cover page width and the line's width,
            #given that the starting "x" coordinate of the cover page is zero (so no need to add it to the result).
            left_x = round((cover_page_width_points - cover_page_font_object.text_length(split_title_string_list[i], fontsize=cover_page_font_size))/2)
            new_text_wrap.append(pos=(left_x, top_y), text=split_title_string_list[i], font=cover_page_font_object, fontsize=cover_page_font_size)
            #The "top_y" coordinate is shifted down by an amount of points equal to the line height without line spacing
            #("line_height_without_line_spacing", which was calculated by subtracting "cover_page_font_size.descender"
            #from "cover_page_font_size.ascender" and then multiplying the result by "cover_page_font_size"), times
            #the line spacing for the cover page ("cover_page_line_spacing").
            top_y += cover_page_line_spacing * line_height_without_line_spacing  

        #The changes are committed to the "cover_page" object.
        new_text_wrap.write_text(cover_page)

        #If a sequence of three subsequent hyphens wasn't found in the original PDF file name,
        #then the "split_author_string_list" would be empty and the "if" statement below wouldn't run.
        if split_author_string_list != []:
            #The "get_cover_page_font_size()" function is called once again for the 
            #author text with the "split_author_string_list" and the optional argument 
            #("cover_title_font_size = cover_page_font_size") to allow the code to
            #set the author name font size to a value no greater than 85% of that
            #of the title font size.
            cover_page_font_size = get_cover_page_font_size(cover_page, cover_page_font_object, split_author_string_list, cover_page_width_points, cover_page_height_points, cover_page_line_spacing, cover_title_font_size = cover_page_font_size)

            #The height of a line of text in the author text font size is 
            #calculated by multiplying the difference between the font object's
            #ascenders and descenders to give the font height at a font size of
            #one, times the cover page font size.
            line_height_without_line_spacing = (cover_page_font_object.ascender - cover_page_font_object.descender) * cover_page_font_size

            #A "TextWriter" object ("new_text_wrap") is used to use the "append()"
            #"TextWriter" method for each line of the author text, with the specified
            #color ("cover_page_color")
            new_text_wrap = pymupdf.TextWriter(cover_page.rect, color=cover_page_color)

            #The author text is to be written directly below the white/black interface at the vertical 
            #center of the page. However, because the "pos" argument of the "append()" method of the PyMuPDF
            #"TextWrap" class draws the text with the baseline "y" coordinate of the text (right above the
            #descender), we then need to move the "top_y" down by the font size itself ("cover_page_font_size", 
            #excluding ascenders and descenders). We also add a vertical spacer "cover_page_vertical_spacer"
            #to avoid the text being too flush with the white/black interface.
            top_y = half_cover_page_height + cover_page_font_size + cover_page_vertical_spacer

            len_split_author_string_list = len(split_author_string_list)
            #Each line in the "split_author_string_list" will be appended to the 
            #initially empty "TextWrap" object "new_text_wrap", with the top-left
            #coordinates specified by the "pos" argument.
            for i in range(len_split_author_string_list):
                #The lines are horizontally centered by halving the difference between the cover page width and the line's width,
                #given that the starting "x" coordinate of the cover page is zero (so no need to add it to the result).
                left_x = round((cover_page_width_points - cover_page_font_object.text_length(split_author_string_list[i], fontsize=cover_page_font_size))/2)
                new_text_wrap.append(pos=(left_x, top_y), text=split_author_string_list[i], font=cover_page_font_object, fontsize=cover_page_font_size)
                #The "top_y" coordinate is shifted down by an amount of points equal to the line height without line spacing
                #("line_height_without_line_spacing", which was calculated by subtracting "cover_page_font_size.descender"
                #from "cover_page_font_size.ascender" and then multiplying the result by "cover_page_font_size"), times
                #the line spacing for the cover page ("cover_page_line_spacing").
                top_y += cover_page_line_spacing * line_height_without_line_spacing  

            #The changes are committed to the "cover_page" object.
            new_text_wrap.write_text(cover_page)

        #The "subset_fonts()" PyMuPDF method will only include the
        #specific characters of the embedded font that were used instead
        #of the entire font, which may save a lot of space, since the font
        #was only used on the cover page.
        doc_cover_page.subset_fonts()

        #Insert the cover page (the page at index zero of "doc_cover_page",
        #"from_page=0, to_page=0" at the very beginning ("start_at=0")
        #of the "doc_output" main document.

        #This will allow the cover page to be in color, while the
        #actual pages of the book will remain in "Grayscale" or
        #"Black and White" format, thus optimizing the file size
        #of the final PDF file.
        doc_output.insert_pdf(doc_cover_page, from_page=0, to_page=0, start_at=0)

    if cumulative_pdf_file_size_estimation > 0:
        output_pdf_file_number += 1

        #The function "get_pdf_file_path" will assemble the current PDF file path from the current
        #working folder ("cwd"), the output folder name, the file name (which may need to be truncated 
        #in order for the path to be under 260 characters) and the output file number.
        pdf_file_path = get_pdf_file_path(cwd, output_folder_name, output_file_name, output_pdf_file_number)

        #Purge hidden metadata and unneeded references
        doc_output.scrub()

        doc_output.save(pdf_file_path,
            garbage=4,          
            deflate=True,       #Compresses uncompressed streams (images, fonts, text)
            use_objstms=True,    #Packs PDF objects into compressed streams (cuts around 25%)
            )
    return set_of_potential_blank_pages, output_pdf_file_number



#The function "generate_pdf_file()" will generate the PDF file.
def generate_pdf_file(json_settings_dictionary, json_default_settings_dictionary, cwd):

    first_page = int(json_settings_dictionary["First Page"])
    #If the value of "first_page" is valid, then it will be zero-indexed by subtracting one from it.
    if is_valid_positive_non_zero_int(first_page):
       first_page -= 1
    #Reset to the default value of 0 (page 1 in zero indexed form) if the JSON data is invalid.
    else:
       first_page = json_default_settings_dictionary["First Page"] - 1     

    last_page = json_settings_dictionary["Last Page"]
    #If the value of "last_page" is valid, then it will be zero-indexed by subtracting one from it.
    if is_valid_positive_non_zero_int(last_page):
       last_page -= 1
    #Reset to the default value of 0 if the JSON data is invalid 
    #(one is not subtracted from it, as zero indicates the last page of the book).
    else:
       last_page = json_default_settings_dictionary["Last Page"]  

    removed_pages_input_string = json_settings_dictionary["Removed Pages"]
    if isinstance(removed_pages_input_string, str):
        #The function "validate_removed_pages()" will validate the inputted list
        #of pages that are to be removed from the original PDF document when
        #generating the new PDF document and return the list of individual
        #pages to be removed, or an empty list (the default value of no 
        #removed pages) if the input string was invalid.
        list_of_individual_removed_pages = validate_removed_pages(removed_pages_input_string)
    else:
        list_of_individual_removed_pages = []

    cover_page_enabled = json_settings_dictionary["Cover Page"]
    #If the value of "is_dark_mode_enabled" isn't a Boolean, then it will
    #be set to its default value of "False".
    if not isinstance(cover_page_enabled, bool):
        cover_page_enabled = json_default_settings_dictionary["Cover Page"]

    cover_page_line_spacing = json_settings_dictionary["Cover Page Line Spacing"]
    #Reset to the default value of 0.9 if the JSON data is invalid (valid value would be a positive int or float).
    if not (is_valid_int_or_float(cover_page_line_spacing) and cover_page_line_spacing > 0):
        cover_page_line_spacing = json_default_settings_dictionary["Cover Page Line Spacing"]

    #The cover page color RGB list is extracted from "json_settings_dictionary"
    #(tuples are stored as Python lists in JSON files).
    cover_page_color_list = json_settings_dictionary["Cover Page Color"]
    #If the list is not comprised of three numbers each between zero and 255 inclusively,
    #then the default setting of [255, 255, 255] will be used instead.
    if not (isinstance(cover_page_color_list, list) and len(cover_page_color_list) == 3 and 
    isinstance(cover_page_color_list[0], int) and cover_page_color_list[0] >= 0 and cover_page_color_list[0] <= 255 and 
    isinstance(cover_page_color_list[1], int) and cover_page_color_list[1] >= 0 and cover_page_color_list[1] <= 255 and 
    isinstance(cover_page_color_list[2], int) and cover_page_color_list[2] >= 0 and cover_page_color_list[2] <= 255):
        cover_page_color_list = json_default_settings_dictionary["Cover Page Color"]
    #Each channel of the RGB tuple is normalized to values between 0 and 1 by dividing it by 255,
    #as PyMuPDF uses values between 0 and 1. Values between 0 and 255 need to be stored in the
    #"json_settings_dictionary" and "json_default_settings_dictionary", as floating point rounding
    #errors could lead to different colors when storing floating point values between 0 and 1.
    cover_page_color = (cover_page_color_list[0]/255, cover_page_color_list[1]/255, cover_page_color_list[2]/255)

    dpi_setting = json_settings_dictionary["DPI Setting"]
    #Reset to the default value of 300 if the JSON data is invalid.
    if not is_valid_positive_non_zero_int(dpi_setting):
        dpi_setting = json_default_settings_dictionary["DPI Setting"]

    max_mb_per_pdf_file = json_settings_dictionary["Maximal File Size"]
    #If the value of "max_mb_per_pdf_file" is valid, then multiply it by a million to get the value in bytes.
    if is_valid_non_negative_int_or_float(max_mb_per_pdf_file):
        max_mb_per_pdf_file = max_mb_per_pdf_file * 1000000
    #Reset to the default value of 100 and multiply the result by a million to get the value in bytes if the JSON data is invalid.
    else:
        max_mb_per_pdf_file = json_default_settings_dictionary["Maximal File Size"] * 1000000

    #"not" is used in front of "json_settings_dictionary["Grayscale Mode"]" in order to
    #give the reverse value of whether or not the "Grayscale Mode" is enabled, as
    #it can only be "Black and White" or "Grayscale Mode".
    black_and_white_mode_enabled = not json_settings_dictionary["Grayscale Mode"]
    #If the value of "black_and_white_mode_enabled" isn't a Boolean, then it will
    #be set to its default value of "True".
    if not isinstance(black_and_white_mode_enabled, bool):
        #"not" is used in front of "json_settings_dictionary["Grayscale Mode"]" in order to
        #give the reverse value of whether or not the "Grayscale Mode" is enabled, as
        #it can only be "Black and White" or "Grayscale Mode".
        black_and_white_mode_enabled = not json_default_settings_dictionary["Grayscale Mode"]

    do_crop_pages = json_settings_dictionary["Auto-Cropping"]
    #If the value of "do_crop_pages" isn't a Boolean, then it will
    #be set to its default value of "True".
    if not isinstance(do_crop_pages, bool):
        do_crop_pages = json_default_settings_dictionary["Auto-Cropping"]

    do_pad_pages = json_settings_dictionary["Auto-Padding"]
    #If the value of "do_crop_pages" isn't a Boolean, then it will
    #be set to its default value of "True".
    if not isinstance(do_pad_pages, bool):
        do_pad_pages = json_default_settings_dictionary["Auto-Padding"]

    horizontal_crop_kernel_size_height_percent = json_settings_dictionary["Left-Right Kernel Size"] 
    #If the value isn't a valid positive integer or float, then the default percentage will be used instead.
    if not is_valid_non_negative_int_or_float(horizontal_crop_kernel_size_height_percent):
        horizontal_crop_kernel_size_height_percent = json_default_settings_dictionary["Left-Right Kernel Size"] / 100
    #The percentage is divided by 100 to give the percentage in decimal format
    horizontal_crop_kernel_size_height_percent /= 100

    horizontal_crop_kernel_radius_kernel_size_percent = json_settings_dictionary["Left-Right Kernel Radius"] 
    #If the value isn't a valid positive integer or float, then the default percentage will be used instead.
    if not is_valid_non_negative_int_or_float(horizontal_crop_kernel_radius_kernel_size_percent):
        horizontal_crop_kernel_radius_kernel_size_percent = json_default_settings_dictionary["Left-Right Kernel Radius"] / 100
    #The percentage is divided by 100 to give the percentage in decimal format
    horizontal_crop_kernel_radius_kernel_size_percent /= 100

    horizontal_crop_margin_buffer_width_percentage = json_settings_dictionary["Left-Right Safe Margin Size"] 
    #If the value isn't a valid positive integer or float, then the default percentage will be used instead.
    if not is_valid_non_negative_int_or_float(horizontal_crop_margin_buffer_width_percentage):
        horizontal_crop_margin_buffer_width_percentage = json_default_settings_dictionary["Left-Right Safe Margin Size"] / 100
    #The percentage is divided by 100 to give the percentage in decimal format
    horizontal_crop_margin_buffer_width_percentage /= 100

    vertical_crop_kernel_size_height_percent = json_settings_dictionary["Top-Bottom Kernel Size"]
    #If the value isn't a valid positive integer or float, then the default percentage will be used instead.
    if not is_valid_non_negative_int_or_float(vertical_crop_kernel_size_height_percent):
        vertical_crop_kernel_size_height_percent = json_default_settings_dictionary["Top-Bottom Kernel Size"] / 100
    #The percentage is divided by 100 to give the percentage in decimal format
    vertical_crop_kernel_size_height_percent /= 100

    vertical_crop_kernel_radius_kernel_size_percent = json_settings_dictionary["Top-Bottom Kernel Radius"] 
    #If the value isn't a valid positive integer or float, then the default percentage will be used instead.
    if not is_valid_non_negative_int_or_float(vertical_crop_kernel_radius_kernel_size_percent):
        vertical_crop_kernel_radius_kernel_size_percent = json_default_settings_dictionary["Top-Bottom Kernel Radius"] / 100
    #The percentage is divided by 100 to give the percentage in decimal format
    vertical_crop_kernel_radius_kernel_size_percent /= 100

    vertical_crop_margin_buffer_height_percentage = json_settings_dictionary["Top-Bottom Safe Margin Size"] 
    #If the value isn't a valid positive integer or float, then the default percentage will be used instead.
    if not is_valid_non_negative_int_or_float(vertical_crop_margin_buffer_height_percentage):
        vertical_crop_margin_buffer_height_percentage = json_default_settings_dictionary["Top-Bottom Safe Margin Size"] / 100
    #The percentage is divided by 100 to give the percentage in decimal format
    vertical_crop_margin_buffer_height_percentage /= 100

    brightness_level = json_settings_dictionary["Initial Brightness Level"]
    #If the value isn't a valid positive integer or float, then the default value will be used instead.
    if not is_valid_non_negative_int_or_float(brightness_level):
        brightness_level = json_default_settings_dictionary["Initial Brightness Level"]

    final_brightness_level = json_settings_dictionary["Final Brightness Level"]
    #If the value isn't a valid positive integer or float, then the default value will be used instead.
    if not is_valid_non_negative_int_or_float(final_brightness_level):
        final_brightness_level = json_default_settings_dictionary["Final Brightness Level"]

    contrast_level = json_settings_dictionary["Initial Contrast Level"]
    #If the value isn't a valid positive integer or float, then the default value will be used instead.
    if not is_valid_non_negative_int_or_float(contrast_level):
        contrast_level = json_default_settings_dictionary["Initial Contrast Level"]

    final_contrast_level = json_settings_dictionary["Final Contrast Level"]
    #If the value isn't a valid positive integer or float, then the default value will be used instead.
    if not is_valid_non_negative_int_or_float(final_contrast_level):
        final_contrast_level = json_default_settings_dictionary["Final Contrast Level"]

    is_dark_mode_enabled = json_settings_dictionary["Dark Mode"]
    #If the value of "is_dark_mode_enabled" isn't a Boolean, then it will
    #be set to its default value of "False".
    if not isinstance(is_dark_mode_enabled, bool):
        is_dark_mode_enabled = json_default_settings_dictionary["Dark Mode"]

    left_margin_width_percent = json_settings_dictionary["Margins Filter Left Margin"]
    #If the value isn't a valid positive integer or float, then the default percentage will be used instead.
    if not is_valid_non_negative_int_or_float(left_margin_width_percent):
        left_margin_width_percent = json_default_settings_dictionary["Margins Filter Left Margin"] / 100
    #The percentage is divided by 100 to give the percentage in decimal format
    left_margin_width_percent /= 100

    right_margin_width_percent = json_settings_dictionary["Margins Filter Right Margin"]
    #If the value isn't a valid positive integer or float, then the default percentage will be used instead.
    if not is_valid_non_negative_int_or_float(right_margin_width_percent):
        right_margin_width_percent = json_default_settings_dictionary["Margins Filter Right Margin"] / 100
    #The percentage is divided by 100 to give the percentage in decimal format
    right_margin_width_percent /= 100

    top_margin_height_percent = json_settings_dictionary["Margins Filter Top Margin"]
    #If the value isn't a valid positive integer or float, then the default percentage will be used instead.
    if not is_valid_non_negative_int_or_float(top_margin_height_percent):
        top_margin_height_percent = json_default_settings_dictionary["Margins Filter Top Margin"] / 100
    #The percentage is divided by 100 to give the percentage in decimal format
    top_margin_height_percent /= 100

    bottom_margin_height_percent = json_settings_dictionary["Margins Filter Bottom Margin"]
    #If the value isn't a valid positive integer or float, then the default percentage will be used instead.
    if not is_valid_non_negative_int_or_float(bottom_margin_height_percent):
        bottom_margin_height_percent = json_default_settings_dictionary["Margins Filter Bottom Margin"] / 100
    #The percentage is divided by 100 to give the percentage in decimal format
    bottom_margin_height_percent /= 100

    do_filter_out_splotches_margins = json_settings_dictionary["Margins Filter"]
    #If the value of "do_filter_out_splotches_margins" isn't a Boolean, then it will
    #be set to its default value of "True".
    if not isinstance(do_filter_out_splotches_margins, bool):
        do_filter_out_splotches_margins = json_default_settings_dictionary["Margins Filter"]

    do_filter_out_splotches_entire_page = json_settings_dictionary["Full-Page Filter"]
    #If the value of "do_filter_out_splotches_entire_page" isn't a Boolean, then it will
    #be set to its default value of "True".
    if not isinstance(do_filter_out_splotches_entire_page, bool):
        do_filter_out_splotches_entire_page = json_default_settings_dictionary["Full-Page Filter"]

    number_of_standard_deviations_for_filtering_page_color_cropping = json_settings_dictionary["Page Color Filter Multiplier When Cropping"]
    #If the value isn't a valid integer or float, then the default value will be used instead.
    if not is_valid_int_or_float(number_of_standard_deviations_for_filtering_page_color_cropping):
        number_of_standard_deviations_for_filtering_page_color_cropping = json_default_settings_dictionary["Page Color Filter Multiplier When Cropping"]   

    number_of_standard_deviations_for_filtering_page_color = json_settings_dictionary["Page Color Filter Multiplier"]
    #If the value isn't a valid integer or float, then the default value will be used instead.
    if not is_valid_int_or_float(number_of_standard_deviations_for_filtering_page_color):
        number_of_standard_deviations_for_filtering_page_color = json_default_settings_dictionary["Page Color Filter Multiplier"]

    number_of_standard_deviations_for_filtering_splotches_margins = json_settings_dictionary["Margins Filter Multiplier"]
    #If the value isn't a valid integer or float, then the default value will be used instead.
    if not is_valid_int_or_float(number_of_standard_deviations_for_filtering_splotches_margins):
        number_of_standard_deviations_for_filtering_splotches_margins = json_default_settings_dictionary["Margins Filter Multiplier"]

    number_of_standard_deviations_for_filtering_splotches_entire_page = json_settings_dictionary["Full-Page Filter Multiplier"]
    #If the value isn't a valid integer or float, then the default value will be used instead.
    if not is_valid_int_or_float(number_of_standard_deviations_for_filtering_splotches_entire_page):
        number_of_standard_deviations_for_filtering_splotches_entire_page = json_default_settings_dictionary["Full-Page Filter Multiplier"]

    pdf_path = os.path.join(cwd, "Original Book PDF File", "*.pdf")
    output_folder_name = "Final Book PDF Files"
    pdf_files = glob.glob(pdf_path)

    for pdf_file_index in range(len(pdf_files)):

        output_pdf_file_number = 0
        cumulative_pdf_file_size_estimation = 0
        #If the cropped width is very narrow (below the threshold "narrow_page_threshold"),
        #then the cropped image will be "widened" (it won't be cropped as much) by an amount
        #equal to "padding_width", which is calculated by halving the difference between either 
        #the average width of the pages of the book (if the length of "list_of_cropped_page_widths"
        #is greater than zero), or the "narrow_page_threshold" and the cropped width, effectively 
        #bringing the width of that cropped page to the average cropped width in the former case, 
        #and to the "narrow_page_threshold" in the latter case.
        list_of_cropped_page_widths = [] 

        #Returns the final component of the path
        file_name_with_extension = os.path.basename(pdf_files[pdf_file_index])
        file_name, extension = os.path.splitext(file_name_with_extension)
        #The "doc" PyMuPDF "Document" object will hold the original PDF document's
        #data at the index "pdf_file_index" of the "pdf_files" list of PDF file paths. 
        doc = pymupdf.open(pdf_files[pdf_file_index])
        #Just to make sure that the first page is positive.
        if first_page < 0:
            first_page = 0
        #If the requested last page is superior to the actual
        #number of pages in the original PDF document, or if
        #it is set to the default value of zero meaning until
        #the last page, the value of "last_page" will be set
        #to the number of pages in the PDF document minus one,
        #as it is zero-indexed.
        number_of_pages = doc.page_count
        if last_page == 0 or last_page > number_of_pages - 1:
            last_page = number_of_pages - 1
        #A new PyMuPDF "Document" object "doc_output" is created
        #to store the information of the final PDF document.
        doc_output = pymupdf.open()

        #The "clear_screen()" function will clear the CLI screen
        #using the appropriate command depending on the operating system.
        clear_screen()

        #The function "get_terminal_dimensions()" will return the number of columns 
        #and rows in the console, to allow to properly format the text and dividers.
        columns, lines = get_terminal_dimensions()

        #The file name is cleaned up, in case the user has formatted it to 
        #generate the cover page.

        #Removing the sequences of three or more hyphens (flanked or not by spaces) 
        #by a space, a hyphen and a space.
        output_file_name = re.sub(r"[ ]*[-]{3,}[ ]*", " - ", file_name)
        #Replacing sequences of at least two consecutive spaces by a single space.
        output_file_name = re.sub(r"[ ]{2,}", " ", output_file_name)

        settings_summary_string = f"Here is the summary of the settings for generating your PDF file '{output_file_name}':\n"
        textwrapped_settings_summary_string = textwrap.fill(settings_summary_string, width=columns)

        color_mode_string = "- Color Mode: "
        if json_settings_dictionary["Grayscale Mode"]:
            color_mode_string += f"Grayscale, Dark Mode {return_on_for_true_and_off_for_false(is_dark_mode_enabled)}"
        else:
            color_mode_string += f"Black and White, Dark Mode {return_on_for_true_and_off_for_false(is_dark_mode_enabled)}"

        if list_of_individual_removed_pages == []:
            removed_pages_string = "- Removed Pages: No Removed Pages"
        else:
            #The function "format_removed_pages_string()" will format the string that will be printed
            #in the menus to indicate which pages from the original PDF will be removed when generating
            #the final PDF document. If the number of characters in the output string exceeds the value
            #of "length_threshold", then the total number of pages removed will be returned in string
            #form (ex: "15 pages removed") instead of a string of all removed pages 
            #(ex: "1-3, 5-10, 12-15, 29, 35")
            removed_pages_string = textwrap.fill(f"- Removed Pages: {format_removed_pages_string(list_of_individual_removed_pages, 100)}", columns)

        #A summary of settings will be printed on-screen:
        # - Cover page ON/OFF (extracted Title and Author if ON)
        # - Max File Size
        # - Color mode (BW vs Grayscale, Dark Mode ON/OFF)
        # - Starting page (if not in list of deleted pages)
        # - End page (if not in list of deleted pages)
        # - List of deleted pages (large threshold to include as much as possible)

        print("=== Settings Summary ===\n\n")

        print(textwrapped_settings_summary_string + "\n")

        print(f"- Cover Page: {return_on_for_true_and_off_for_false(cover_page_enabled)}")

        #The lists "split_title_string_list" and "split_author_string_list",
        #which are used when generating the cover page, need to be initialized
        #as empty lists in case the user does not include a cover page, as the
        #function call "save_pdf()" includes these as arguments.
        split_title_string_list = []
        split_author_string_list = []

        #If the "Cover Page" mode is enabled, then the title and author name will
        #automatically be extracted from the file name, where a divider of at least
        #three consecutive hyphens between the title and author name, and carriage
        #return locations indicated by two or more successive spaces
        #("ex: 'Title  Subtitle --- Author Name'").
        if cover_page_enabled:

            print(f"  Cover Page Line Spacing: {json_settings_dictionary["Cover Page Line Spacing"]}")

            #The function "get_cover_page_color_string()" will retrieve the color string corresponding to the
            #tuple of the chosen color in "colors_dict". If the color tuple isn't in "colors_dict", then it
            #means that the user has provided a custom color, so its RGB and Hex code information will be
            #returned in string form instead.
            print(textwrap.fill(f"  Cover Page Color: {get_cover_page_color_string(json_settings_dictionary)}", width=columns))

            #As the cover page preview will be centered along the
            #full width of the console window, the function 
            #"get_terminal_dimensions()" is not called, as
            #it returns 75% of the full width of the console.

            #Detect columns (width) and lines (height)
            #Returns a named tuple; default fallback is (80, 24)
            size = shutil.get_terminal_size(fallback=(80, 24))
            full_width_columns, lines = int(size.columns), int(size.lines)

            #The "Cover Page Preview" heading is printed on-screen.
            print("  Cover Page Preview:\n")

            #The file name is split along sequences of three or more successive
            #hyphens, flanked by zero or more spaces.
            split_file_name = re.split(r"[ ]*[-]{3,}[ ]*", file_name)

            #If the length of the split file name is greater or equal to two,
            #then it means that a title and author name or title, subtitle and author name
            #have been provided and will be split 
            if len(split_file_name) >= 2:
                title_string = split_file_name[0]
                author_string = split_file_name[1]
            #If the length of the split file name is equal to one,
            #then it means that there are no sequences of three or more
            #successive hyphens and there are probably no author names in 
            #the title, so "author_string" will be set to "None"
            else:
                title_string = split_file_name[0]
                author_string = None
                split_author_string_list = []

            #The title is split along sequences of two or more successive
            #spaces denoting the location of carriage returns.

            #The "split_title_author_string_for_carriage_returns()"
            #function will split a title page string along sequences 
            #of two or more successive spaces denoting the location 
            #of carriage returns. It will allow for one or more
            #successive carriage returns (four spaces for two
            #carriage returns, for example). It returns the
            #split string, with the "\n" characters added to their
            #appropriate places and the extra spaces trimmed out.
            split_title_string_list = split_title_author_string_for_carriage_returns(title_string)

            if author_string != None:
                #The author name is split along sequences of two or more successive
                #spaces denoting the location of carriage returns.
                split_author_string_list = split_title_author_string_for_carriage_returns(author_string)

            #Each line of the split title string will be 
            #printed with centered alignment.
            for line in split_title_string_list:
                print(line.center(full_width_columns))
            #A one-line spacer is included between the
            #title and author name, if present.
            if author_string != None:
                #A separator of three hyphens mimics the 
                #black/white interface on the cover page.
                print("---".center(full_width_columns))
                #Each line of the split author name will be
                #printed with centered alignment
                for line in split_author_string_list:
                    print(line.center(full_width_columns))
            #A one-line spacer is included after the 
            #cover page preview.
            print("")

        #As the value was zero-indexed, +1 must be added 
        #to it when displaying the page number to the user,
        #or comparing it to the values in the list of removed
        #pages.
        if first_page + 1 not in list_of_individual_removed_pages:

            print(f"- First Page: {first_page + 1}")
        #As the value was zero-indexed, +1 must be added 
        #to it when displaying the page number to the user,
        #or comparing it to the values in the list of removed
        #pages.
        if last_page + 1 not in list_of_individual_removed_pages:
            #The function "get_last_page_string()" will return "Last Page of Original PDF"
            #if the current "Last Page" setting is set to zero, and the string version of
            #"json_settings_dictionary["Last Page"]" otherwise.
            print(f"- Last Page: {get_last_page_string(json_settings_dictionary)}")

        print(removed_pages_string)

        print(f"- Max File Size: {round(max_mb_per_pdf_file/1000000)} MB")

        print(color_mode_string)

        print(f"- Auto-Cropping: {return_on_for_true_and_off_for_false(json_settings_dictionary["Auto-Cropping"])}")

        print(f"- Auto-Padding: {return_on_for_true_and_off_for_false(json_settings_dictionary["Auto-Padding"])}")

        print("\n")

        start_time = time.perf_counter()
        #The previous estimation of the remaining number of seconds is stored in the variable
        #"previoous_estimated_seconds" and will be used instead of the current calculation
        #if it exceeds the previous estimation, so as to avoid the ETA timer increasing 
        #its estimation.
        previous_estimated_seconds = 0

        #The "set_of_potential_blank_pages" gathers all the page number in the original 
        #document whose pages are likely blank pages, either because they do not contain
        #enough non-white pixels to be cropped, or because they are horizontally cropped
        #very significantly in such a way that their width/height ratio becomes much 
        #smaller than the average text page's aspect ratio. These pages will be listed
        #on-screen for the user to add them to the list of removed pages.
        set_of_potential_blank_pages = set()
        #The "list_of_original_document_page_numbers" will tally up the list of
        #all the page numbers in the original PDF document that were included in
        #"doc_output". This will allow to determine which cropped pages were 
        #horizontally cropped much more than other pages and are thus likely
        #blank pages.
        list_of_original_document_page_numbers = []

        #As "last_page" is zero-indexed, +1 needs to be added,
        #in order to include it in the range.
        for page_index in range(first_page, last_page + 1):

            #As "page_index" is zero-indexed, +1 needs to be added to it when comparing
            #with the first value of "list_of_individual_removed_pages"
            if list_of_individual_removed_pages == [] or (list_of_individual_removed_pages != [] and page_index + 1 != list_of_individual_removed_pages[0]): 

                if (cumulative_pdf_file_size_estimation >= max_mb_per_pdf_file):

                    #The "save_pdf()" function will generate a cover page (if the "Cover Page" mode is enabled)
                    #and output the PyMuPDF "Document" object as a PDF file with the corresponding output PDF
                    #file number in parentheses (e.g.,: "Book Title - Subtitle by Author Name (Part 1).pdf").
                    set_of_potential_blank_pages, output_pdf_file_number = save_pdf(cwd,
                    doc_output,
                    list_of_original_document_page_numbers,
                    is_dark_mode_enabled,
                    do_crop_pages,
                    do_pad_pages,
                    cover_page_enabled,
                    cover_page_color,
                    cover_page_line_spacing,
                    split_title_string_list,
                    split_author_string_list,
                    cumulative_pdf_file_size_estimation,
                    output_pdf_file_number,
                    output_folder_name,
                    output_file_name, 
                    set_of_potential_blank_pages)

                    #A new "doc_output" PyMuPDF "Document" object
                    #is instantiated to continue generating the 
                    #final PDF files.
                    doc_output = pymupdf.open()
                    #The value of "cumulative_pdf_file_size_estimation"
                    #is reset to zero megabytes, as this is a new PDF file.
                    cumulative_pdf_file_size_estimation = 0

                #The "process_image()" function will extract the image file from the
                #PDF document as a grayscale Pixmap object, convert it to a NumPy array 
                #to process it, and then convert the NumPy array back to a Pixmap object,
                #which will be included in the "doc_output" Document object.
                (doc_output,
                cumulative_pdf_file_size_estimation,  
                list_of_cropped_page_widths, 
                set_of_potential_blank_pages,
                list_of_original_document_page_numbers) = process_image(
                    doc,
                    doc_output,
                    page_index, 
                    dpi_setting,
                    cumulative_pdf_file_size_estimation,
                    do_filter_out_splotches_margins,
                    number_of_standard_deviations_for_filtering_page_color_cropping,
                    number_of_standard_deviations_for_filtering_page_color,
                    number_of_standard_deviations_for_filtering_splotches_margins,
                    do_filter_out_splotches_entire_page,
                    number_of_standard_deviations_for_filtering_splotches_entire_page,
                    black_and_white_mode_enabled,
                    do_crop_pages,        
                    horizontal_crop_kernel_size_height_percent,
                    horizontal_crop_kernel_radius_kernel_size_percent,
                    horizontal_crop_margin_buffer_width_percentage,
                    vertical_crop_kernel_size_height_percent,
                    vertical_crop_kernel_radius_kernel_size_percent,
                    vertical_crop_margin_buffer_height_percentage,
                    brightness_level,
                    final_brightness_level,
                    contrast_level,
                    final_contrast_level,
                    is_dark_mode_enabled,
                    left_margin_width_percent,
                    right_margin_width_percent,
                    top_margin_height_percent,
                    bottom_margin_height_percent,
                    list_of_cropped_page_widths,
                    set_of_potential_blank_pages,
                    list_of_original_document_page_numbers
                    )

                #The function "display_progress" will display the progress string in the console
                #and return the estimated number of seconds for the code to complete.

                #The previous estimation of the remaining number of seconds is stored in the variable
                #"previoous_estimated_seconds" and will be used instead of the current calculation
                #if it exceeds the previous estimation, so as to avoid the ETA timer increasing 
                #its estimation.
                previous_estimated_seconds = display_progress(page_index, first_page, last_page, start_time, previous_estimated_seconds, list_of_individual_removed_pages)

            #As "page_index" is zero-indexed, +1 needs to be added to it when comparing
            #with the first value of "list_of_individual_removed_pages"

            #Once the removed page has been dealt with, it will be popped out of 
            #the "list_of_individual_removed_pages" list.
            if list_of_individual_removed_pages != [] and page_index + 1 >= list_of_individual_removed_pages[0]:
                list_of_individual_removed_pages.pop(0)

        #The "save_pdf()" function will generate a cover page (if the "Cover Page" mode is enabled)
        #and output the PyMuPDF "Document" object as a PDF file with the corresponding output PDF
        #file number in parentheses (e.g.,: "Book Title - Subtitle by Author Name (Part 1).pdf").
        set_of_potential_blank_pages, output_pdf_file_number = save_pdf(cwd,
        doc_output,
        list_of_original_document_page_numbers,
        is_dark_mode_enabled,
        do_crop_pages,
        do_pad_pages,
        cover_page_enabled,
        cover_page_color,
        cover_page_line_spacing,
        split_title_string_list,
        split_author_string_list,
        cumulative_pdf_file_size_estimation,
        output_pdf_file_number,
        output_folder_name,
        output_file_name,
        set_of_potential_blank_pages)  

        #At the end of the code
        doc_output.close()
        doc.close()

        #The "set_of_potential_blank_pages" gathers all the page number in the original 
        #document whose pages are likely blank pages, either because they do not contain
        #enough non-white pixels to be cropped, or because they are horizontally cropped
        #very significantly in such a way that their width/height ratio becomes much 
        #smaller than the average text page's aspect ratio. These pages will be listed
        #on-screen for the user to add them to the list of removed pages.
        if set_of_potential_blank_pages != set():
            #The function "get_terminal_dimensions()" will return the number of columns 
            #and rows in the console, to allow to properly format the text and dividers.
            columns, lines = get_terminal_dimensions()
            #The set of potential blank pages is converted into a list and sorted.
            list_of_potential_blank_pages = sorted(list(set_of_potential_blank_pages))
            #Should there be more than one element in the list "set_of_potential_blank_pages",
            #then each of its elements will be converted into strings through the "map" method. 
            if len(set_of_potential_blank_pages) > 1:
                list_of_potential_blank_page_strings = list(map(str, list_of_potential_blank_pages))
                #Each element is then joined by a comma and a space to form a comma-separated
                #string of potentially blank pages.
                string_of_potential_blank_pages = ", ".join(list_of_potential_blank_page_strings)
            #Otherwise the single element will be indexed at the index zero and converted into a string.
            else:
                string_of_potential_blank_pages = str(list_of_potential_blank_pages[0])
            #As the user may wish to run the code again with the added removed pages found in the
            #list of potentially blank pages, the data from the current removed pages and the 
            #list of potentially blank pages will be combined.
            formatted_string_of_additional_removed_pages = ""
            removed_pages_input_string = json_settings_dictionary["Removed Pages"]
            if isinstance(removed_pages_input_string, str):
                #The function "validate_removed_pages()" will validate the inputted list
                #of pages that are to be removed from the original PDF document when
                #generating the new PDF document and return the list of individual
                #pages to be removed, or an empty list (the default value of no 
                #removed pages) if the input string was invalid.
                list_of_individual_removed_pages = validate_removed_pages(removed_pages_input_string)

                if list_of_individual_removed_pages != []:

                    list_of_individual_removed_pages += list_of_potential_blank_pages
                    #The function "format_removed_pages_string()" will format the string that will be printed
                    #in the menus to indicate which pages from the original PDF will be removed when generating
                    #the final PDF document. If the number of characters in the output string exceeds the value
                    #of "length_threshold", then the total number of pages removed will be returned in string
                    #form (ex: "15 pages removed") instead of a string of all removed pages 
                    #(ex: "1-3, 5-10, 12-15, 29, 35")

                    #Here we sort "list_of_individual_removed_pages", as we have concatenated two lists
                    #and we want to ensure the numbers in the final string are presented in ascending order.

                    #A 1000000-character limit is used before which the number of removed pages will be displayed instead 
                    #of the complete list (ensuring that all of the removed pages are printed on-screen).
                    formatted_string_of_additional_removed_pages = format_removed_pages_string(sorted(list_of_individual_removed_pages), 1000000)

            potential_blank_page_list_f_string = f"The following pages are potentially blank pages that you could add to the list of removed pages to ensure good auto-padding results: {string_of_potential_blank_pages}"
            blocked_potential_blank_page_list_string = textwrap.fill(potential_blank_page_list_f_string, width=columns)
            print("")
            print(blocked_potential_blank_page_list_string)

            if formatted_string_of_additional_removed_pages != "":
                print("")
                additional_removed_pages_f_string = f"Here is the adjusted list of removed pages with these potential blank pages added to it: {formatted_string_of_additional_removed_pages}"
                blocked_additional_removed_pages_f_string = textwrap.fill(additional_removed_pages_f_string, width=columns)
                print(blocked_additional_removed_pages_f_string)

        print("")
        input("Your PDF has successfully been generated! Press any key to continue.")
        return json_settings_dictionary


#The function "validate_removed_pages()" will validate the inputted list
#of pages that are to be removed from the original PDF document when
#generating the new PDF document and return the list of individual
#pages to be removed, or an empty list (the default value of no 
#removed pages) if the input string was invalid.
def validate_removed_pages(removed_pages_input_string):

    #The "span_start_end_pages" must be instantiated as an empty
    #list, in case there are no spans in the starting string,
    #as there is a "if span_start_end_pages != []:" conditional
    #statement further down in the code.
    span_start_end_pages = []

    #If there is a hyphen in "removed_pages_input_string", then we need to look for sequences of
    #two digits connected by a hyphen, with or without spaces between the digits and the hyphens.
    if "-" in removed_pages_input_string:
        #A list is made with the resulting iterators, as we need to iterate over them more
        #than once to extract the spans and the comma- or space-separated individual digits.
        span_matches = list(re.finditer(r"(\d+)[ ]*-[ ]*(\d+)", removed_pages_input_string))
        #The start and end indices for each match will be stored in "span_start_end_indices"
        #for slicing out the spans from "removed_pages_input_string".
        span_start_end_indices = [[match.start(), match.end()] for match in span_matches]
        #The spans are split along the hyphens and space separators and the resulting list
        #of starting and ending digits in every spans are stored in the list "span_start_end_pages"
        #(ex: [["9", "14"], ["5", "1"]] from the starting string "9 - 14, 7, 5-1").
        span_start_end_pages = [re.split(r"[ ]*-[ ]*", match.group(0)) for match in span_matches]
        #The digit characters within each sublist are converted into integers and then sorted
        #(ex: [[9, 14], [1, 5]] from the starting list [["9", "14"], ["5", "1"]]).
        span_start_end_pages = [sorted([int(sublist[0]), int(sublist[1])]) for sublist in span_start_end_pages]
        #The sublists are then sorted along their first digits ("key=lambda x: x[0]")
        #(ex: [[1, 5], [9, 14]] from the starting list [[9, 14], [1, 5]]).
        span_start_end_pages = sorted(span_start_end_pages, key=lambda x: x[0])

        #The input string "removed_pages_input_string" is cycled over in reverse order to avoid
        #indexing issues while slicing out the spans. Each span is sliced out by accessing its
        #starting and ending index within the "span_start_end_indices" list of span start-end 
        #indices list (ex: [[span1_start, span1_end], [span2_start, span2_end]]).
        for i in range(len(span_start_end_indices)-1, -1, -1):
            removed_pages_input_string = removed_pages_input_string[:span_start_end_indices[i][0]] + removed_pages_input_string[span_start_end_indices[i][1]:]

    #Any invalid characters that are not commas, spaces or digits in the 
    #sliced string will then trigger an "invalid input" prompt. 
    is_string_valid = re.search(r"[^, \d]", removed_pages_input_string) == None
    #If the sliced string is valid, then 
    if is_string_valid:
        #if "," in removed_pages_input_string, then the string will be split along those commas
        individual_pages_without_spans = [element.strip() for element in removed_pages_input_string.split(",") if element not in ["", " "]]
        #The resulting elements are split along spaces, in case the user separated some or all
        #of their numbers with spaces instead of commas.
        individual_pages_without_spans = [element.split(" ") for element in individual_pages_without_spans]
        #The nested list (the "split" method results in nested lists) is 
        #flattened and only non empty ('element != ""') elements are retained.
        individual_pages_without_spans = [element for sublist in individual_pages_without_spans for element in sublist if element != ""]
        #Each element is then converted from a string form to an int form.
        individual_pages_without_spans = [int(element) for element in individual_pages_without_spans]
        #The list of ints is sorted in case the user didn't provide them in order.
        individual_pages_without_spans = sorted([int(element) for element in individual_pages_without_spans])
        #If there were spans in the original string, then these will be expanded
        #into lists of individual pages within the span, and the resulting nested
        #list will be flattened to have all of the individual removed pages within
        #the spans, which will then be concatenated with the list of individual 
        #pages without spans, and converted into a set to remove duplicates, and
        #finally sorted. 
        if span_start_end_pages != []:
            #The spans are converted into "range" objects and then lists of all the pages within the spans.
            span_pages_individual_numbers = [list(range(span[0], span[1]+1)) for span in span_start_end_pages]
            #The nested list of span lists is flattened.
            span_pages_individual_numbers = [element for sublist in span_pages_individual_numbers for element in sublist]
            #The list of individual pages and spans are concatenated, duplicates are removed with "set()", the set is 
            #converted back into a list, which is then sorted to give the final list of pages to be removed.
            list_of_individual_removed_pages = sorted(list(set(individual_pages_without_spans + span_pages_individual_numbers)))
        else:
            #The list of individual pages and spans are concatenated, duplicates are removed with "set()", the set is 
            #converted back into a list, which is then sorted to give the final list of pages to be removed.
            list_of_individual_removed_pages = sorted(list(set(individual_pages_without_spans)))

        return list_of_individual_removed_pages
    #If the input string of comma-separated pages to remove
    #is invalid (it contains characters other than spaces, commas
    #and digits after extracting the page spans) then an empty list
    #will be returned, which is the default value of no removed pages.
    else:
        return []


#The function "format_removed_pages_string()" will format the string that will be printed
#in the menus to indicate which pages from the original PDF will be removed when generating
#the final PDF document. If the number of characters in the output string exceeds the value
#of "length_threshold", then the total number of pages removed will be returned in string
#form (ex: "15 pages removed") instead of a string of all removed pages 
#(ex: "1-3, 5-10, 12-15, 29, 35")
def format_removed_pages_string(list_of_individual_removed_pages, length_threshold = 20):

    #The "is_in_span" Boolean flag, initialized to "False",
    #will be set to "True" when the current page number when
    #cycling over the pages numbers in the list
    #"list_of_individual_removed_pages" is found
    #in a span (the next number equals that number 
    #plus one), and "False" otherwise. This will
    #prevent the numbers that aren't the span
    #boundaries from being printed in the output
    #string ("output_string").
    is_in_span = False
    output_string = ""
    len_list_of_individual_removed_pages = len(list_of_individual_removed_pages)
    for i in range(len_list_of_individual_removed_pages):
        #If "is_in_span" is "False" (not currently in a span), and the number of 
        #pages in the list "list_of_individual_removed_pages" is greater than one
        #(as you need at least two pages to form a span) and the current page index
        #is lower than the last page index (as you need to look at the page after
        #the current one), and if value of the page number after the current one
        #is one unit greater than the current page value at the index "i",
        #then it means that a new span has just begun. The value of "is_in_span"
        #will be set to "True" and the "output_string" will be appended with the
        #current page number, followed by a hyphen. 
        if (not is_in_span and len_list_of_individual_removed_pages > 1 and 
        i < len_list_of_individual_removed_pages - 1 and 
        list_of_individual_removed_pages[i+1] == list_of_individual_removed_pages[i] + 1):
            output_string += str(list_of_individual_removed_pages[i]) + "-"
            is_in_span = True
        #If you are currently in a span and you are not yet at the last page index
        #and the next page number is not exactly one unit above the current page,
        #then it means that you have reached the final page in the span. The value
        #of "is_in_span" is set to "False" and the final number of the span is added
        #to "output_string", followed by a comma and a space (as you are not yet at
        #the last page number).
        elif (is_in_span and len_list_of_individual_removed_pages > 1 and 
        i < len_list_of_individual_removed_pages - 1 and 
        list_of_individual_removed_pages[i+1] != list_of_individual_removed_pages[i] + 1):
            output_string += str(list_of_individual_removed_pages[i]) + ", "
            is_in_span = False
        #If you are currently in a span and you have reached the final page, then it 
        #means that you can include the final page in the span, this time without a
        #comma after it, as you have reached the final page index in the list
        #"list_of_individual_removed_pages".
        elif (is_in_span and i == len_list_of_individual_removed_pages - 1):
            output_string += str(list_of_individual_removed_pages[i])
            is_in_span = False
        #If you are not in a span, then the page number is added to 
        #the "output_string", with a comma and a space after it if 
        #this is not the last index in "list_of_individual_removed_pages". 
        elif not is_in_span:
            output_string += str(list_of_individual_removed_pages[i]) + (", ") * (i != len_list_of_individual_removed_pages - 1)

    #The length of the "output_string" is measured,
    #and if it exceeds the "length_threshold", the
    #number of removed pages will be returned instead
    #of all of the removed pages in span form.
    len_output_string = len(output_string)
    if len_output_string > length_threshold:
        return f"{len(list_of_individual_removed_pages)} pages removed"
    else:
        return output_string

#The function "load_json_data()" will load the JSON data from file
#and store them in the "json_settings_dictionary", or initialize the
#dictionary based on the values of "json_default_settings_dictionary".
def load_json_data(json_settings_file_path_name):

    json_default_settings_dictionary = {
            "_comment_1" : first_page_comment_stirng,
            "First Page" : 1,

            "_comment_2" : last_page_comment_stirng,
            "Last Page" : 0,

            "_comment_3" : removed_pages_comment_string,
            "Removed Pages" : "",

            "_comment_4" : cover_page_mode_comment_string,
            "Cover Page" : True,

            "_comment_5" : cover_page_line_spacing_comment_string,
            "Cover Page Line Spacing" : 0.9,

            "_comment_6" : cover_page_color_selection_comment_string,
            "Cover Page Color" : [255, 255, 255],

            "_comment_7" : dpi_setting_comment_string,
            "DPI Setting" : 300,

            "_comment_8" : max_mb_per_pdf_file_comment_string,
            "Maximal File Size" : 100.0,

            "_comment_9" : grayscale_mode_enabled_comment_string,
            "Grayscale Mode" : True,

            "_comment_10" : auto_cropping_comment_string,
            "Auto-Cropping" : True,

            "_comment_11" : auto_padding_mode_comment_string,
            "Auto-Padding" : True,

            "_comment_12" : horizontal_crop_kernel_size_height_percent_comment_string,
            "Left-Right Kernel Size" : 2.0,

            "_comment_13" : horizontal_crop_kernel_radius_kernel_size_percent_comment_string,
            "Left-Right Kernel Radius" : 30.0,

            "_comment_14" : horizontal_crop_margin_buffer_width_percentage_comment_string,
            "Left-Right Safe Margin Size" : 1.5,

            "_comment_15" : vertical_crop_kernel_size_height_percent_comment_string,
            "Top-Bottom Kernel Size" : 8.0,

            "_comment_16" : vertical_crop_kernel_radius_kernel_size_percent_comment_string,
            "Top-Bottom Kernel Radius" : 20.0,

            "_comment_17" : vertical_crop_margin_buffer_height_percentage_comment_string,
            "Top-Bottom Safe Margin Size" : 2.0,

            "_comment_18" : initial_brightness_level_comment_string,
            "Initial Brightness Level" : 1.0,

            "_comment_19" : final_brightness_level_comment_string,
            "Final Brightness Level" : 1.0,

            "_comment_20" : initial_contrast_level_comment_string,
            "Initial Contrast Level" : 1.0,

            "_comment_21" : final_contrast_level_comment_string,
            "Final Contrast Level" : 1.0,

            "_comment_22" : dark_mode_comment_string,
            "Dark Mode" : False,

            "_comment_23" : left_margin_width_percent_comment_string,
            "Margins Filter Left Margin" : 2.5,

            "_comment_24" : right_margin_width_percent_comment_string,
            "Margins Filter Right Margin" : 2.5,

            "_comment_25" : top_margin_height_percent_comment_string,
            "Margins Filter Top Margin" : 2.5,

            "_comment_26" : bottom_margin_height_percent_comment_string,
            "Margins Filter Bottom Margin" : 2.5,

            "_comment_27" : do_filter_out_splotches_margins_comment_string,
            "Margins Filter" : True,

            "_comment_28" : do_filter_out_splotches_entire_page_comment_string,
            "Full-Page Filter" : True,

            "_comment_29" : number_of_standard_deviations_for_filtering_page_color_when_cropping_comment_string,
            "Page Color Filter Multiplier When Cropping": -1.5,

            "_comment_30" : number_of_standard_deviations_for_filtering_page_color_comment_string,
            "Page Color Filter Multiplier" : 0.0,

            "_comment_31" : number_of_standard_deviations_for_filtering_splotches_margins_comment_string,
            "Margins Filter Multiplier" : -0.25,

            "_comment_32" : number_of_standard_deviations_for_filtering_splotches_entire_page_comment_string,
            "Full-Page Filter Multiplier" : 3.0
        }

    need_to_generate_new_json_file = False
    if os.path.isfile(json_settings_file_path_name):
        #A try-except statement is used in case
        #the JSON file is malformed or empty,
        #in which case the Boolean variable
        #"need_to_generate_new_json_file" will
        #be set to "True" and the "if" statement
        #below this one would run.
        try:
            #The "utf-8-sig" encoding handles files with or without a BOM automatically
            with open(json_settings_file_path_name, "r", encoding="utf-8-sig") as f:
                json_settings_dictionary = json.load(f)
            #If the user has manually entered zero as the value for the 
            #"Removed Pages" key of the JSON file, it will be changed to
            #an empty string, which will be replaced by "No Removed Pages"
            #when displaying the "Removed Pages" setting on-screen.
            if json_settings_dictionary["Removed Pages"] == 0:
                json_settings_dictionary["Removed Pages"] = ""
                print("UPDATED DICT!")
                #The function "atomic_save()" will create a temporary JSON file with the updated changes.
                #If the files is created successfully, then the files will be swapped. If a problem is 
                #encountered, the temp file will be unlinked and an error log will be reported.
                atomic_save(json_settings_dictionary, json_settings_file_path_name) 

        except json.JSONDecodeError:
            need_to_generate_new_json_file = True
    else:
        need_to_generate_new_json_file = True

    if need_to_generate_new_json_file:
        #A deep copy (since it contains a list of deleted pages) of 
        #"json_default_settings_dictionary" is made so as to avoid having
        #both "json_settings_dictionary" and "json_default_settings_dictionary"
        #pointing to the same address.
        json_settings_dictionary = copy.deepcopy(json_default_settings_dictionary)

        #Create a low-level file descriptor (used for atomic saves)
        #The two access flags "os.O_RDWR" and "os.O_CREAT" allow for the file to be 
        #read and written to and created if it doesn't already exist, respectively.
        file_descriptor = os.open(json_settings_file_path_name, os.O_RDWR | os.O_CREAT)
        with os.fdopen(file_descriptor, "w+", encoding="utf-8") as f:
            #Write the default values found in "json_settings_dictionary" in the empty JSON file, 
            #with four space indentations to make it more human-readable.
            json.dump(json_settings_dictionary, f, indent=4)
            #Ensure the data is flushed to hardware.
            f.flush()
            #"os.fsync(f.fileno())" is required to force the OS to physically commit
            #every bit of information to the hardware storage right now, preventing 
            #a situation where an empty file might be created if the computer crashed
            #before the OS finished waiting before committing the file to memory. 
            os.fsync(f.fileno())
    return json_default_settings_dictionary, json_settings_dictionary


#The function "textwrap_action_strings_in_menu_action_dict()", which takes in 
#a menu action dictionary comprised of one character keys and values made up
#of a three-member tuple (action string, function, function arguments).
#The action strings ("value[0]") will be textwrapped and the modified
#dictionary will be returned.
def textwrap_action_strings_in_menu_action_dict(menu_action_dict):

    #The function "get_terminal_dimensions()" will return the number of columns 
    #and rows in the console, to allow to properly format the text and dividers.
    columns, lines = get_terminal_dimensions()

    for key, value in menu_action_dict.items():
        value[0] = textwrap.fill(value[0], columns)
    return menu_action_dict


#The function "quit_function()" will call
#"sys.exit()" with the exit code "1" meaning
#"success".
def quit_function():
    sys.exit(1)


#The function "back_to_main_menu_function()"
#will set the Boolean flags "is_in_submenu" and 
#"is_in_sub_submenu" to "False", which will break the submenu
#"while" loops and return to the main menu.
def back_to_main_menu_function(json_settings_dictionary):
    global is_in_submenu 
    global is_in_sub_submenu
    global is_in_sub_sub_submenu
    is_in_submenu = False
    is_in_sub_submenu = False
    is_in_sub_sub_submenu = False    
    return json_settings_dictionary


def back_to_submenu_function(json_settings_dictionary):
    global is_in_sub_submenu
    is_in_sub_submenu = False  
    return json_settings_dictionary


#The "invalid_menu_choice()" function will be called when the
#user enters invalid input in one of the functions called by
#the "run_menu()" function.
def invalid_menu_choice(json_settings_dictionary):
    input("\nInvalid choice, press any key to continue.")
    return json_settings_dictionary


#The function "run_menu" will retrieve and call the function
#at the appropriate choice key in the "menu_actions_dict"
def run_menu(menu_actions_dict, json_settings_dictionary):

    for key, (label, _, _) in menu_actions_dict.items():
        print(f"[{key}] {label}")

    choice = input("\nSelect an option: ").strip().lower()

    #In case the user just pressed "Enter",
    #"json_settings_dictionary" will be returned
    #(no action, this will avoid an error message).
    if choice == "":
        return json_settings_dictionary

    #If you can successfully access the "menu_actions_dict" dictionary
    #with the value of "choice", you then have access to the tuple containing
    #(function label, function, args). Indexing the tuple at the position one
    #gives the function itself, and indexing it at the position 2 gives you the
    #arguments for that function as a list, which must be unpacked with the "*" operator. 
    nested_list = menu_actions_dict.get(choice, [None, invalid_menu_choice, (json_settings_dictionary,)]) 
    return nested_list[1](*nested_list[2])


#The "reset_to_default_setting()" function will reset the setting to its default value
#found while accessing the value of the "json_default_settings_dictionary" dictionary 
#with the key "setting_label_key".
def reset_to_default_setting(setting_label_key, json_settings_dictionary, 
json_default_settings_dictionary, json_settings_file_path_name):
    json_settings_dictionary[setting_label_key] = json_default_settings_dictionary[setting_label_key]
    #The function "atomic_save()" will create a temporary JSON file with the updated changes.
    #If the files is created successfully, then the files will be swapped. If a problem is 
    #encountered, the temp file will be unlinked and an error log will be reported.
    atomic_save(json_settings_dictionary, json_settings_file_path_name)   
    return json_settings_dictionary


#The "set_to_true_false()" function will set the Boolean setting found while accessing
#the "json_settings_dictionary" dictionary with the key "setting_label_key" to the provided
#value ("boolean_value"). 
def set_to_true_false(boolean_value, setting_label_key, json_settings_dictionary, 
json_settings_file_path_name):
    json_settings_dictionary[setting_label_key] = boolean_value
    #The function "atomic_save()" will create a temporary JSON file with the updated changes.
    #If the files is created successfully, then the files will be swapped. If a problem is 
    #encountered, the temp file will be unlinked and an error log will be reported.
    atomic_save(json_settings_dictionary, json_settings_file_path_name)    
    return json_settings_dictionary


#The "toggle_boolean_setting()" function will set the Boolean setting found while accessing
#the "json_settings_dictionary" dictionary with the key "setting_label_key" to the opposite
#value of the current setting ("True" if the current setting is "False" and vice-versa). 
def toggle_boolean_setting(setting_label_key, json_settings_dictionary, 
json_settings_file_path_name):
    if json_settings_dictionary[setting_label_key]:
        json_settings_dictionary[setting_label_key] = False
    else:
        json_settings_dictionary[setting_label_key] = True 
    #The function "atomic_save()" will create a temporary JSON file with the updated changes.
    #If the files is created successfully, then the files will be swapped. If a problem is 
    #encountered, the temp file will be unlinked and an error log will be reported.
    atomic_save(json_settings_dictionary, json_settings_file_path_name)    
    return json_settings_dictionary


#The "set_numeric_setting()" function will set the value of the setting found while accessing
#the "json_settings_dictionary" dictionary with the key "setting_label_key" to the provided
#value ("setting_value"). 
def set_numeric_setting(setting_value, setting_label_key, json_settings_dictionary, 
json_settings_file_path_name):
    json_settings_dictionary[setting_label_key] = setting_value
    #The function "atomic_save()" will create a temporary JSON file with the updated changes.
    #If the files is created successfully, then the files will be swapped. If a problem is 
    #encountered, the temp file will be unlinked and an error log will be reported.
    atomic_save(json_settings_dictionary, json_settings_file_path_name)   
    return json_settings_dictionary


#The "get_removed_pages_setting_string_for_menus()" will return "No Removed Pages"
#if no pages have been removed, and the string of removed pages returned by calling
#the "format_removed_pages_string()" function on the returned value of 
#"validate_removed_pages()" otherwise.
def get_removed_pages_setting_string_for_menus(json_settings_dictionary, columns):
    no_removed_pages_string = "No Removed Pages"
    #The formatted list of currently removed pages is obtained by calling the "format_removed_pages_string()"
    #function on the returned list from the "validate_removed_pages()" function, which was called with the
    #value of 'json_settings_dictionary["Removed Pages"] with a 1000000-character limit before which the 
    #number of removed pages will be displayed instead of the complete list (ensuring that all of the 
    #removed pages are printed on-screen).
    current_removed_pages_string = textwrap.fill(format_removed_pages_string(validate_removed_pages(json_settings_dictionary["Removed Pages"]), 1000000), width=columns)
    #If no pages are removed (empty string), then the "no_removed_pages_string"
    #will be displayed on-screen instead.
    if current_removed_pages_string == "":
        current_removed_pages_string = no_removed_pages_string
    return current_removed_pages_string


#The "set_color_mode()" function will set the color mode of the final PDF document (Black and White vs Grayscale).
def set_color_mode(json_settings_dictionary, json_default_settings_dictionary, json_settings_file_path_name):

    global is_in_submenu
    is_in_submenu = True

    #The "while is_in_submenu" loop will continue
    #running until the user returns to the main menu
    while is_in_submenu:

        #The "clear_screen()" function will clear the CLI screen
        #using the appropriate command depending on the operating system.
        clear_screen()

        color_mode_menu_actions_dict = {
        "1": ["Enable the 'Grayscale Mode' (Recommended. Better quality text)", set_to_true_false, (True, "Grayscale Mode", 
            json_settings_dictionary, json_settings_file_path_name)],
        "2": ["Enable the 'Black and White Mode' (Smaller file sizes)", set_to_true_false, (False, "Grayscale Mode", 
            json_settings_dictionary, json_settings_file_path_name)],
        "r": ["Reset to the Default Setting", reset_to_default_setting, ("Grayscale Mode", 
        json_settings_dictionary, json_default_settings_dictionary, json_settings_file_path_name)],
        "t": ["Toggle Dark Mode On/Off (Writes light text on dark pages)", toggle_boolean_setting, ("Dark Mode", json_settings_dictionary, 
            json_settings_file_path_name)],
        "m": ["Main Menu", back_to_main_menu_function, (json_settings_dictionary,)],
        "q": ["Quit", quit_function, ()]}

        #The function "textwrap_action_strings_in_menu_action_dict()", which takes in 
        #a menu action dictionary comprised of one character keys and values made up
        #of a three-member tuple (action string, function, function arguments).
        #The action strings ("value[0]") will be textwrapped and the modified
        #dictionary will be returned.
        color_mode_menu_actions_dict = textwrap_action_strings_in_menu_action_dict(color_mode_menu_actions_dict)

        print("=== Set Color Mode ===\n\n")

        #The function "get_terminal_dimensions()" will return the number of columns 
        #and rows in the console, to allow to properly format the text and dividers.
        columns, lines = get_terminal_dimensions()
        #As the default color mode setting is already printed on-screen, the " (default setting: True)"
        #portion of the instructions string is removed.
        textwrapped_instructions_string = textwrap.fill(grayscale_mode_enabled_comment_string.replace(" (default setting: True)", ""), width=columns)

        if (json_settings_dictionary["Dark Mode"]):
            print("Dark mode is currently turned ON.\n")
        else:
            print("Dark mode is currently turned OFF (Default value).\n")

        if (json_settings_dictionary["Grayscale Mode"]):
            current_setting = "Grayscale Mode"
        else:
            current_setting = "Black and White Mode"

        print(f"Current Setting: {current_setting} | Default: Grayscale Mode\n")

        print(textwrapped_instructions_string + "\n")

        #The function "run_menu" will retrieve and call the function
        #at the appropriate choice key in the "menu_actions_dict"
        json_settings_dictionary = run_menu(color_mode_menu_actions_dict, json_settings_dictionary)
    return json_settings_dictionary


#The "set_first_page_number()" function will set the first page of the original PDF document that will
#be included in the final PDF file.
def set_first_page_number(json_settings_dictionary, json_default_settings_dictionary, json_settings_file_path_name):

    global is_in_sub_submenu
    is_in_sub_submenu = True

    while is_in_sub_submenu:
        try:
            #The "clear_screen()" function will clear the CLI screen
            #using the appropriate command depending on the operating system.
            clear_screen()

            print("=== Set First Page ===\n\n")

            #The function "get_terminal_dimensions()" will return the number of columns 
            #and rows in the console, to allow to properly format the text and dividers.
            columns, lines = get_terminal_dimensions()

            print(f"Current Setting: {json_settings_dictionary["First Page"]} | Default: {json_default_settings_dictionary["First Page"]}.\n")

            textwrapped_toggle_string = textwrap.fill(cover_page_mode_comment_string, width=columns)

            textwrapped_instructions_string = textwrap.fill(first_page_comment_stirng, width=columns)
            textwrapped_input_string = textwrap.fill("Enter the first page number (1 or higher), or select one of the above options:", width=columns)

            print(textwrapped_instructions_string + " ")
            print(f"\n[r] Reset to the Default Setting\n[b] Page Management Menu\n[m] Main Menu\n[q] Quit\n")
            choice = input(textwrapped_input_string + " ").strip().lower()

            if choice == "":
                #A continue needs to be used, as we don't want 
                #the code below the "elif" statements to run,
                #which would cause a ValueError on int("").
                continue
            elif choice == "m":
                #The function "back_to_main_menu_function()"
                #will set the Boolean flags "is_in_submenu" and 
                #"is_in_sub_submenu" to "False", which will break the submenu
                #"while" loops and return to the main menu.
                back_to_main_menu_function(json_settings_dictionary)
                #A continue needs to be used, as we don't want 
                #the code below the "elif" statements to run,
                #which would cause a ValueError on int("m").
                continue
            elif choice == "b":
                is_in_sub_submenu = False
                #A continue needs to be used, as we don't want 
                #the code below the "elif" statements to run,
                #which would cause a ValueError on int("b").
                continue
            elif choice == "r":
                #The "reset_to_default_setting()" function will reset the setting to its default value
                #found while accessing the value of the "json_default_settings_dictionary" dictionary 
                #with the key "setting_label_key". 
                #The "True" return value (if used) will break the submenu
                #loop and allow to return to the main menu or
                #the nested menu.
                json_settings_dictionary = reset_to_default_setting("First Page", json_settings_dictionary, 
                    json_default_settings_dictionary, json_settings_file_path_name)
                #A continue needs to be used, as we don't want 
                #the code below the "elif" statements to run,
                #which would cause a ValueError on int("r").
                continue
            elif choice == "q":
                quit_function()
            first_page = int(choice)
            if first_page > 0:
                #The "set_numeric_setting()" function will set the value of the setting found while accessing
                #the "json_settings_dictionary" dictionary with the key "setting_label_key" to the provided
                #value ("setting_value"). 
                json_settings_dictionary = set_numeric_setting(first_page, "First Page", json_settings_dictionary, 
                json_settings_file_path_name)
            else:
                input("\nInvalid choice, press any key to continue.") 
        except ValueError:
            input("\nInvalid choice, press any key to continue.")
    return json_settings_dictionary


#The "set_last_page_number()" function will set the last page of the original PDF document that will
#be included in the final PDF file.
def set_last_page_number(json_settings_dictionary, json_default_settings_dictionary, json_settings_file_path_name):

    global is_in_sub_submenu
    is_in_sub_submenu = True

    while is_in_sub_submenu:
        try:
            #The "clear_screen()" function will clear the CLI screen
            #using the appropriate command depending on the operating system.
            clear_screen()

            print("=== Set Last Page ===\n\n")

            #The function "get_terminal_dimensions()" will return the number of columns 
            #and rows in the console, to allow to properly format the text and dividers.
            columns, lines = get_terminal_dimensions()

            #The function "get_last_page_string()" will return "Last Page of Original PDF"
            #if the current "Last Page" setting is set to zero, and the string version of
            #"json_settings_dictionary["Last Page"]" otherwise.
            print(textwrap.fill(f"Current Setting: {get_last_page_string(json_settings_dictionary)} | Default: Last Page of Original PDF.", width=columns) + "\n")

            textwrapped_toggle_string = textwrap.fill(cover_page_mode_comment_string, width=columns)

            textwrapped_instructions_string = textwrap.fill(last_page_comment_stirng, width=columns)
            textwrapped_input_string = textwrap.fill("Enter the last page number (1 or higher, or '0' to include all pages), or select one of the above options:", width=columns)

            print(textwrapped_instructions_string)
            print(f"\n[r] Reset to the Default Setting\n[b] Page Management Menu\n[m] Main Menu\n[q] Quit\n")
            choice = input(textwrapped_input_string + " ").strip().lower()

            if choice == "":
                #A continue needs to be used, as we don't want 
                #the code below the "elif" statements to run,
                #which would cause a ValueError on int("").
                continue
            elif choice == "m":
                #The function "back_to_main_menu_function()"
                #will set the Boolean flags "is_in_submenu" and 
                #"is_in_sub_submenu" to "False", which will break the submenu
                #"while" loops and return to the main menu.
                back_to_main_menu_function(json_settings_dictionary)
                #A continue needs to be used, as we don't want 
                #the code below the "elif" statements to run,
                #which would cause a ValueError on int("m").
                continue
            elif choice == "b":
                is_in_sub_submenu = False
                #A continue needs to be used, as we don't want 
                #the code below the "elif" statements to run,
                #which would cause a ValueError on int("b").
                continue
            elif choice == "r":
                #The "reset_to_default_setting()" function will reset the setting to its default value
                #found while accessing the value of the "json_default_settings_dictionary" dictionary 
                #with the key "setting_label_key". 
                #The "True" return value (if used) will break the submenu
                #loop and allow to return to the main menu or
                #the nested menu.
                json_settings_dictionary = reset_to_default_setting("Last Page", json_settings_dictionary, 
                    json_default_settings_dictionary, json_settings_file_path_name)
                #A continue needs to be used, as we don't want 
                #the code below the "elif" statements to run,
                #which would cause a ValueError on int("r").
                continue
            elif choice == "q":
                quit_function()
            last_page = int(choice)
            if last_page >= 0 and (last_page == 0 or last_page >= json_settings_dictionary["First Page"]):
                #The "set_numeric_setting()" function will set the value of the setting found while accessing
                #the "json_settings_dictionary" dictionary with the key "setting_label_key" to the provided
                #value ("setting_value"). 
                json_settings_dictionary = set_numeric_setting(last_page, "Last Page", json_settings_dictionary, 
                json_settings_file_path_name)
            elif last_page != 0 and last_page < json_settings_dictionary["First Page"]:
                last_page_smaller_than_first_page_error_string = "\n" + textwrap.fill(f"Please enter a 'Last Page' number that is at least the value of the 'First Page' number of {json_settings_dictionary["First Page"]}. Press any key to continue.", width=columns)
                input(last_page_smaller_than_first_page_error_string)
            else:
                input("\nInvalid choice, press any key to continue.")  
        except ValueError:
            input("\nInvalid choice, press any key to continue.")
    return json_settings_dictionary


#The "set_removed_pages()" function will allow the user to input a comma-separated string of 
#page numbers from the original PDF document that are to be removed from the final PDF document.
def set_removed_pages(json_settings_dictionary, json_default_settings_dictionary, json_settings_file_path_name):

    global is_in_sub_submenu
    is_in_sub_submenu = True

    while is_in_sub_submenu:
        try:
            #The "clear_screen()" function will clear the CLI screen
            #using the appropriate command depending on the operating system.
            clear_screen()

            print("=== Set Removed Pages ===\n\n")

            #The function "get_terminal_dimensions()" will return the number of columns 
            #and rows in the console, to allow to properly format the text and dividers.
            columns, lines = get_terminal_dimensions()

            textwrapped_instructions_string = textwrap.fill(removed_pages_comment_string, width=columns)

            textwrapped_input_string = textwrap.fill(f"Enter the pages to remove (e.g., 1, 3, 5-10), type '0' to include all pages, or select one of the above options:", width=columns)

            print(f"Current Setting: {get_removed_pages_setting_string_for_menus(json_settings_dictionary, columns)} | Default: No Removed Pages.\n")

            print(textwrapped_instructions_string)

            print(f"\n[r] Include All Pages (Reset Removed Pages)\n[b] Page Management Menu\n[m] Main Menu\n[q] Quit\n")

            choice = input(textwrapped_input_string + " ").strip().lower()

            if choice == "":
                #A continue needs to be used, as we don't want 
                #the code below the "elif" statements to run,
                #which would cause a ValueError on int("").
                continue
            elif choice == "m":
                #The function "back_to_main_menu_function()"
                #will set the Boolean flags "is_in_submenu" and 
                #"is_in_sub_submenu" to "False", which will break the submenu
                #"while" loops and return to the main menu.
                back_to_main_menu_function(json_settings_dictionary)
                #A continue needs to be used, as we don't want 
                #the code below the "elif" statements to run,
                #which would cause a ValueError on int("m").
                continue
            elif choice == "b":
                is_in_sub_submenu = False
                #A continue needs to be used, as we don't want 
                #the code below the "elif" statements to run,
                #which would cause a ValueError on int("b").
                continue
            #If the user has input "0" or "r", then the
            #list of removed pages will be reset to
            #its default value of an empty string.
            elif choice in ["0", "r"]:
                #The "reset_to_default_setting()" function will reset the setting to its default value
                #found while accessing the value of the "json_default_settings_dictionary" dictionary 
                #with the key "setting_label_key". 
                #The "True" return value (if used) will break the submenu
                #loop and allow to return to the main menu or
                #the nested menu.
                json_settings_dictionary = reset_to_default_setting("Removed Pages", json_settings_dictionary, 
                    json_default_settings_dictionary, json_settings_file_path_name)
                #A continue needs to be used, as we don't want 
                #the code below the "elif" statements to run,
                #which would cause a ValueError on int("r").
                continue
            elif choice == "q":
                quit_function()

            #The function "validate_removed_pages()" will validate the inputted list
            #of pages that are to be removed from the original PDF document when
            #generating the new PDF document and return the list of individual
            #pages to be removed, or an empty list (the default value of no 
            #removed pages) if the input string was invalid.
            list_of_individual_removed_pages = validate_removed_pages(choice)
            if list_of_individual_removed_pages != []:
                #The function "format_removed_pages_string()" will format the string that will be printed
                #in the menus to indicate which pages from the original PDF will be removed when generating
                #the final PDF document. If the number of characters in the output string exceeds the value
                #of "length_threshold", then the total number of pages removed will be returned in string
                #form (ex: "15 pages removed") instead of a string of all removed pages 
                #(ex: "1-3, 5-10, 12-15, 29, 35"). Here, a threshold of 1000000 characters
                #is used to ensure that all of the removed pages are included.
                json_settings_dictionary["Removed Pages"] = format_removed_pages_string(list_of_individual_removed_pages, 1000000)
                #The function "atomic_save()" will create a temporary JSON file with the updated changes.
                #If the files is created successfully, then the files will be swapped. If a problem is 
                #encountered, the temp file will be unlinked and an error log will be reported.
                atomic_save(json_settings_dictionary, json_settings_file_path_name)

            else:
                input("\nInvalid choice, press any key to continue.")   
        except ValueError:
            input("\nInvalid choice, press any key to continue.")
    return json_settings_dictionary


#The "set_removed_pages()" function will allow the user to set the line spacing of the cover page text.
def set_cover_page_line_spacing(json_settings_dictionary, json_default_settings_dictionary, json_settings_file_path_name):

    global is_in_sub_sub_submenu
    is_in_sub_sub_submenu = True

    while is_in_sub_sub_submenu:
        try:
            #The "clear_screen()" function will clear the CLI screen
            #using the appropriate command depending on the operating system.
            clear_screen()

            print("=== Set Cover Page Line Spacing ===\n\n")

            #The function "get_terminal_dimensions()" will return the number of columns 
            #and rows in the console, to allow to properly format the text and dividers.
            columns, lines = get_terminal_dimensions()

            textwrapped_toggle_string = textwrap.fill(cover_page_mode_comment_string, width=columns)

            textwrapped_instructions_string = textwrap.fill(cover_page_line_spacing_comment_string, width=columns)

            textwrapped_input_string = textwrap.fill(f"Enter the cover page line spacing (over 0.0), or select one of the above options:", width=columns)

            if (json_settings_dictionary["Cover Page"]):
                cover_page_state = "ON"
            else:
                cover_page_state = "OFF"

            print(f"Cover page is currently turned {cover_page_state}{" (Default value)" * json_settings_dictionary["Cover Page"]}.\n")

            print(f"Current Setting: {json_settings_dictionary["Cover Page Line Spacing"]} | Default: {json_default_settings_dictionary["Cover Page Line Spacing"]}.\n")

            print(textwrapped_toggle_string + "\n")
            print(textwrapped_instructions_string)

            print(f"\n[t] Toggle Cover Page On/Off\n[r] Reset to the Default Setting\n[b] Cover Page Menu\n[m] Main Menu\n[q] Quit\n")

            choice = input(textwrapped_input_string + " ").strip().lower()

            if choice == "":
                #A continue needs to be used, as we don't want 
                #the code below the "elif" statements to run,
                #which would cause a ValueError on int("").
                continue
            elif choice == "m":
                #The function "back_to_main_menu_function()"
                #will set the Boolean flags "is_in_submenu" and 
                #"is_in_sub_submenu" to "False", which will break the submenu
                #"while" loops and return to the main menu.
                back_to_main_menu_function(json_settings_dictionary)
                #A continue needs to be used, as we don't want 
                #the code below the "elif" statements to run,
                #which would cause a ValueError on int("m").
                continue
            elif choice == "b":
                is_in_sub_sub_submenu = False
                #A continue needs to be used, as we don't want 
                #the code below the "elif" statements to run,
                #which would cause a ValueError on int("b").
                continue
            elif choice == "r":
                #The "reset_to_default_setting()" function will reset the setting to its default value
                #found while accessing the value of the "json_default_settings_dictionary" dictionary 
                #with the key "setting_label_key". 
                #The "True" return value (if used) will break the submenu
                #loop and allow to return to the main menu or
                #the nested menu.
                json_settings_dictionary = reset_to_default_setting("Cover Page Line Spacing", json_settings_dictionary, 
                    json_default_settings_dictionary, json_settings_file_path_name)
                #A continue needs to be used, as we don't want 
                #the code below the "elif" statements to run,
                #which would cause a ValueError on int("r").
                continue
            elif choice == "t":
                #The "toggle_boolean_setting()" function will set the Boolean setting found while accessing
                #the "json_settings_dictionary" dictionary with the key "setting_label_key" to the opposite
                #value of the current setting ("True" if the current setting is "False" and vice-versa). 

                #The outcome string will be returned from the "toggle_boolean_setting()" 
                #function all if the default parameter Boolean flag "do_not_print_results" 
                #is set to "True", to ensure that the same state of the Filter is printed 
                #both at the top of the screen (current state) and in the confirmation string.
                json_settings_dictionary = toggle_boolean_setting("Cover Page", json_settings_dictionary, 
                    json_settings_file_path_name)
                #A continue needs to be used, as we don't want 
                #the code below the "elif" statements to run,
                #which would cause a ValueError on int("r").
                continue
            elif choice == "q":
                quit_function()

            cover_page_line_spacing = float(choice)
            if cover_page_line_spacing > 0:
                #The "set_numeric_setting()" function will set the value of the setting found while accessing
                #the "json_settings_dictionary" dictionary with the key "setting_label_key" to the provided
                #value ("setting_value"). 
                json_settings_dictionary = set_numeric_setting(cover_page_line_spacing, "Cover Page Line Spacing", json_settings_dictionary, 
                json_settings_file_path_name)
            else:
                input("\nInvalid choice, press any key to continue.")   
        except ValueError:
            input("\nInvalid choice, press any key to continue.")
    return json_settings_dictionary


#The "set_cover_page_color()" function will set the light color of the cover page.
def set_cover_page_color(json_settings_dictionary, json_default_settings_dictionary, json_settings_file_path_name):

    global is_in_sub_sub_submenu
    is_in_sub_sub_submenu = True

    while is_in_sub_sub_submenu:
        try:
            #The "clear_screen()" function will clear the CLI screen
            #using the appropriate command depending on the operating system.
            clear_screen()

            print("=== Set Cover Page Color ===\n\n")

            #The function "get_terminal_dimensions()" will return the number of columns 
            #and rows in the console, to allow to properly format the text and dividers.
            columns, lines = get_terminal_dimensions()

            textwrapped_toggle_string = textwrap.fill(cover_page_mode_comment_string, width=columns)

            textwrapped_instructions_string = textwrap.fill(cover_page_color_selection_comment_string, width=columns)

            textwrapped_input_string = textwrap.fill(f"Enter the cover page RGB color (e.g., '0, 255, 255' for Cyan) or hex code (e.g., '#00FFFF' for Cyan), or select one of the above options:", width=columns)

            if (json_settings_dictionary["Cover Page"]):
                cover_page_state = "ON"
            else:
                cover_page_state = "OFF"

            print(f"Cover page is currently turned {cover_page_state}{" (Default value)" * json_settings_dictionary["Cover Page"]}.\n")

            #The function "get_cover_page_color_string()" will retrieve the color string corresponding to the
            #tuple of the chosen color in "colors_dict". If the color tuple isn't in "colors_dict", then it
            #means that the user has provided a custom color, so its RGB and Hex code information will be
            #returned in string form instead.
            print(textwrap.fill(f"Current Setting: {get_cover_page_color_string(json_settings_dictionary)} | Default: {colors_dict[tuple(json_default_settings_dictionary["Cover Page Color"])]}.", width=columns) + "\n")

            print(textwrapped_toggle_string + "\n")
            print(textwrapped_instructions_string + "\n")

            #All of the preset color options in "colors_dict" are printed on-screen.
            for index, value in enumerate(colors_dict.values(), start=1):
                print(f"[{index}] {value}")

            print(f"[t] Toggle Cover Page On/Off\n[r] Reset to the Default Setting\n[b] Cover Page Menu\n[m] Main Menu\n[q] Quit\n")

            choice = input(textwrapped_input_string + " ").strip().lower()

            if choice == "":
                #A continue needs to be used, as we don't want 
                #the code below the "elif" statements to run,
                #which would cause a ValueError on int("").
                continue
            elif choice == "m":
                #The function "back_to_main_menu_function()"
                #will set the Boolean flags "is_in_submenu" and 
                #"is_in_sub_submenu" to "False", which will break the submenu
                #"while" loops and return to the main menu.
                back_to_main_menu_function(json_settings_dictionary)
                #A continue needs to be used, as we don't want 
                #the code below the "elif" statements to run,
                #which would cause a ValueError on int("m").
                continue
            elif choice == "b":
                is_in_sub_sub_submenu = False
                #A continue needs to be used, as we don't want 
                #the code below the "elif" statements to run,
                #which would cause a ValueError on int("b").
                continue
            elif choice == "r":
                #The "reset_to_default_setting()" function will reset the setting to its default value
                #found while accessing the value of the "json_default_settings_dictionary" dictionary 
                #with the key "setting_label_key". 
                #The "True" return value (if used) will break the submenu
                #loop and allow to return to the main menu or
                #the nested menu.
                json_settings_dictionary = reset_to_default_setting("Cover Page Color", json_settings_dictionary, 
                    json_default_settings_dictionary, json_settings_file_path_name)
                #A continue needs to be used, as we don't want 
                #the code below the "elif" statements to run,
                #which would cause a ValueError on int("r").
                continue
            elif choice == "t":
                #The "toggle_boolean_setting()" function will set the Boolean setting found while accessing
                #the "json_settings_dictionary" dictionary with the key "setting_label_key" to the opposite
                #value of the current setting ("True" if the current setting is "False" and vice-versa). 

                #The outcome string will be returned from the "toggle_boolean_setting()" 
                #function all if the default parameter Boolean flag "do_not_print_results" 
                #is set to "True", to ensure that the same state of the Filter is printed 
                #both at the top of the screen (current state) and in the confirmation string.
                json_settings_dictionary = toggle_boolean_setting("Cover Page", json_settings_dictionary, 
                    json_settings_file_path_name)
                #A continue needs to be used, as we don't want 
                #the code below the "elif" statements to run,
                #which would cause a ValueError on int("r").
                continue
            elif choice == "q":
                quit_function()
            #If there is a comma in "choice", it likely means that
            #the user has inputted a comma-separated list of RGB values.
            if "," in choice:
                #A search result with a pattern of three sequences of or more digits 
                #, each separated from one another by a comma and zero or more spaces  
                #is captured in the "rgb_search_result" variable. If the result wasn't 
                #"None", then the "if" statement below would run. 
                rgb_search_result = re.search(r"\d+,[ ]*\d+,[ ]*\d+", choice)
                if rgb_search_result:
                    #The string of the search result's group zero (entire string) is
                    #split along sequences of a comma and zero or more spaces, leaving
                    #behind the RGB values in string form, which are stored in the
                    #"rgb_list" list of strings.
                    split_rgb_string = re.split(r",[ ]*", rgb_search_result.group(0))
                    #Each string is converted into an integer and the resulting "map"
                    #object is stored in the "rgb_list" as a list of integers.
                    rgb_list = list(map(int, split_rgb_string))
                    #If all of the RGB values are at most 255 (they wouldn't be negative
                    #because the pattern didn't include any hyphens), then the "rgb_list"
                    #is stored in "json_settings_dictionary["Cover Page Color"]".
                    if all(number <= 255 for number in rgb_list):
                        json_settings_dictionary["Cover Page Color"] = rgb_list
                        #The function "atomic_save()" will create a temporary JSON file with the updated changes.
                        #If the files is created successfully, then the files will be swapped. If a problem is 
                        #encountered, the temp file will be unlinked and an error log will be reported.
                        atomic_save(json_settings_dictionary, json_settings_file_path_name)
                    else:
                        input("\nInvalid choice, press any key to continue.")
                else:
                    input("\nInvalid choice, press any key to continue.")  
            #If the user has input a hexadecimal string, it would likely be at least 3 characters
            #- 3-digit shorthand where each character is doubled (e.g., "F0C" becomes "FF00CC", which in turn becomes (255, 0, 204)),
            #- 4-digit shorthand including the alpha channel at the end (e.g., "F0C00" becomes "FF00CC00", which in turn becomes (255, 0, 204, 0)),
            #- 6-digit normal hex code,
            #- 8-digit hex code with 2 alpha digits at the end (which are ignored by this code) 
            elif isinstance(choice, str) and len(choice) >= 3:
                #A search result with a pattern of three or more consecutive characters 
                #that are either a digit 0-9 or a letter among A-F (upper or lowercase) 
                #is captured in the "hex_search_result" variable. If the result wasn't 
                #"None", then the "if" statement below would run. 
                hex_search_result = re.search(r"[a-fA-F0-9]{3,}", choice)
                if hex_search_result:
                    #The string from the group zero of "hex_search_result" is stored in "hex_string"
                    hex_string = hex_search_result.group(0)
                    #If a 3- or 4-digit shorthand hex form is used, the three first 
                    #digits will be duplicated (ignoring the fourth alpha digit, 
                    #if present).
                    if len(hex_string) in [3, 4]:
                        hex_string = 2 * hex_string[0] + 2 * hex_string[1] + 2 * hex_string[2] 
                    #The "hex_string" is sliced in three sections of two characters starting
                    #at the index zero and each two-character slice is casted to an integer
                    #in base 16, giving the RGB values that are stored in the list "rgb_list".
                    rgb_list = [int(hex_string[0:2], 16), int(hex_string[2:4], 16), int(hex_string[4:6], 16)]
                    json_settings_dictionary["Cover Page Color"] = rgb_list
                    #The function "atomic_save()" will create a temporary JSON file with the updated changes.
                    #If the files is created successfully, then the files will be swapped. If a problem is 
                    #encountered, the temp file will be unlinked and an error log will be reported.
                    atomic_save(json_settings_dictionary, json_settings_file_path_name)
                else:
                    input("\nInvalid choice, press any key to continue.")  
            #Otherwise, the user might have entered a single digit to select
            #one of the preset color options.
            else:
                #If the number is an integer (otherwise a ValueError would have been raised,
                #and the "except" statement would run), and its value is within the number 
                #of choices (+1 is added to "len(colors_dict)", as the starting point in the
                #range is 1), the list of RGB values corresponding to the preset color option
                #will be stored in "json_settings_dictionary["Cover Page Color"]". 
                choice = int(choice)
                if choice in range(1, len(colors_dict) + 1):
                    #The RGB values are stored as a list and not a tuple, as the JSON file 
                    #can only store JSON arrays, which are closely related to Python lists. 
                    #The "colors_dict.keys()" view needs to be converted into a list 
                    #in order to be indexed with "choice-1", hence the 
                    #"list(colors_dict.keys())[choice-1]".
                    json_settings_dictionary["Cover Page Color"] = list(list(colors_dict.keys())[choice-1])
                    #The function "atomic_save()" will create a temporary JSON file with the updated changes.
                    #If the files is created successfully, then the files will be swapped. If a problem is 
                    #encountered, the temp file will be unlinked and an error log will be reported.
                    atomic_save(json_settings_dictionary, json_settings_file_path_name)
                else:
                    input("\nInvalid choice, press any key to continue.")
        except ValueError:
            input("\nInvalid choice, press any key to continue.")
    return json_settings_dictionary


#The "cover_page_menu()" function will run a "while is_in_submenu"
#loop that will allow the user to navigate the menu, and the loop will 
#be broken out of when they select the "Quit" option.
def cover_page_menu(json_settings_dictionary, json_default_settings_dictionary, json_settings_file_path_name):

    global is_in_sub_submenu
    is_in_sub_submenu = True

    while is_in_sub_submenu:
        #The "clear_screen()" function will clear the CLI screen
        #using the appropriate command depending on the operating system.
        clear_screen()

        #The function "get_terminal_dimensions()" will return the number of columns 
        #and rows in the console, to allow to properly format the text and dividers.
        columns, lines = get_terminal_dimensions()

        textwrapped_toggle_string = textwrap.fill(cover_page_mode_comment_string, width=columns)

        if (json_settings_dictionary["Cover Page"]):
            cover_page_state = "ON"
        else:
            cover_page_state = "OFF"

        cover_page_menu_actions_dict = {   
        "1": [f"Set Cover Page Line Spacing        ({json_settings_dictionary["Cover Page Line Spacing"]})", set_cover_page_line_spacing, (json_settings_dictionary, json_default_settings_dictionary, json_settings_file_path_name)],
        #tuples are stored as Python lists in JSON files
        "2": [f"Set Cover Page Color               ({get_cover_page_color_string(json_settings_dictionary)})", set_cover_page_color, (json_settings_dictionary, json_default_settings_dictionary, json_settings_file_path_name)],
        "t": [f"Toggle Cover Page On/Off           ({cover_page_state})", toggle_boolean_setting, ("Cover Page", json_settings_dictionary, json_settings_file_path_name)],
        "b": ["Page Management Menu", back_to_submenu_function, (json_settings_dictionary,)],
        "m": ["Main Menu", back_to_main_menu_function, (json_settings_dictionary,)],
        "q": ["Quit", quit_function, ()]}

        #The function "textwrap_action_strings_in_menu_action_dict()", which takes in 
        #a menu action dictionary comprised of one character keys and values made up
        #of a three-member tuple (action string, function, function arguments).
        #The action strings ("value[0]") will be textwrapped and the modified
        #dictionary will be returned.
        cover_page_menu_actions_dict = textwrap_action_strings_in_menu_action_dict(cover_page_menu_actions_dict)

        print("=== Cover Page Menu ===\n\n")

        print(f"Cover page is currently turned {cover_page_state}{" (Default value)" * json_settings_dictionary["Cover Page"]}.\n")

        print(textwrapped_toggle_string + "\n")

        #The function "run_menu" will retrieve and call the function
        #at the appropriate choice key in the "menu_actions_dict"
        json_settings_dictionary = run_menu(cover_page_menu_actions_dict, json_settings_dictionary)
    return json_settings_dictionary


#The "page_management_menu()" function will display the 'Page Range' submenu that will allow the user
#to set the first and last pages from the original PDF that will be included in the final PDF document.
def page_management_menu(json_settings_dictionary, json_default_settings_dictionary, json_settings_file_path_name):

    global is_in_submenu
    is_in_submenu = True

    while is_in_submenu:

        #The function "get_terminal_dimensions()" will return the number of columns 
        #and rows in the console, to allow to properly format the text and dividers.
        columns, lines = get_terminal_dimensions()

        textwrapped_instructions_string = textwrap.fill(cover_page_mode_comment_string, width=columns)

        if (json_settings_dictionary["Cover Page"]):
            cover_page_state = "Cover Page ON"
        else:
            cover_page_state = "Cover Page OFF"

        page_management_menu_actions_dict = {
        "1": [f"Set First Page Number        ({json_settings_dictionary["First Page"]})", set_first_page_number, (json_settings_dictionary, json_default_settings_dictionary, json_settings_file_path_name)],
        "2": [f"Set Last Page Number         ({get_last_page_string(json_settings_dictionary)})", set_last_page_number, (json_settings_dictionary, json_default_settings_dictionary, json_settings_file_path_name)],
        "3": [f"Set Removed Pages            ({get_removed_pages_setting_string_for_menus(json_settings_dictionary, columns)})", set_removed_pages, (json_settings_dictionary, json_default_settings_dictionary, json_settings_file_path_name)],
        "4": [f"Cover Page Menu              ({cover_page_state})", cover_page_menu, (json_settings_dictionary, json_default_settings_dictionary, json_settings_file_path_name)],
        "m": ["Main Menu", back_to_main_menu_function, (json_settings_dictionary,)],
        "q": ["Quit", quit_function, ()]}

        #The function "textwrap_action_strings_in_menu_action_dict()", which takes in 
        #a menu action dictionary comprised of one character keys and values made up
        #of a three-member tuple (action string, function, function arguments).
        #The action strings ("value[0]") will be textwrapped and the modified
        #dictionary will be returned.
        page_management_menu_actions_dict = textwrap_action_strings_in_menu_action_dict(page_management_menu_actions_dict)

        #The "clear_screen()" function will clear the CLI screen
        #using the appropriate command depending on the operating system.
        clear_screen()

        print("=== Page Management Menu ===\n\n")

        #The function "run_menu" will retrieve and call the function
        #at the appropriate choice key in the "menu_actions_dict"
        json_settings_dictionary = run_menu(page_management_menu_actions_dict, json_settings_dictionary)
    return json_settings_dictionary


#The "set_dpi()" function will set the DPI of the images extracted from the original PDF document.
def set_dpi(json_settings_dictionary, json_default_settings_dictionary, json_settings_file_path_name):

    global is_in_submenu
    is_in_submenu = True

    while is_in_submenu:
        try:
            #The "clear_screen()" function will clear the CLI screen
            #using the appropriate command depending on the operating system.
            clear_screen()

            print("=== Set DPI ===\n\n")

            #The function "get_terminal_dimensions()" will return the number of columns 
            #and rows in the console, to allow to properly format the text and dividers.
            columns, lines = get_terminal_dimensions()

            print(f"Current Setting: {json_settings_dictionary["DPI Setting"]} DPI | Default: {json_default_settings_dictionary["DPI Setting"]} DPI.\n")

            textwrapped_instructions_string = textwrap.fill(dpi_setting_comment_string, width=columns)
            textwrapped_input_string = textwrap.fill(f"Enter the DPI setting (50-600 DPI), or select one of the above options:", width=columns)

            print(textwrapped_instructions_string)
            print(f"\n[r] Reset to the Default Setting\n[m] Main Menu\n[q] Quit\n")
            choice = input(textwrapped_input_string + " ").strip().lower()

            if choice == "":
                #A continue needs to be used, as we don't want 
                #the code below the "elif" statements to run,
                #which would cause a ValueError on int("").
                continue
            elif choice == "m":
                #The function "back_to_main_menu_function()"
                #will set the Boolean flags "is_in_submenu" and 
                #"is_in_sub_submenu" to "False", which will break the submenu
                #"while" loops and return to the main menu.
                back_to_main_menu_function(json_settings_dictionary)
                #A continue needs to be used, as we don't want 
                #the code below the "elif" statements to run,
                #which would cause a ValueError on int("m").
                continue
            elif choice == "r":
                #The "reset_to_default_setting()" function will reset the setting to its default value
                #found while accessing the value of the "json_default_settings_dictionary" dictionary 
                #with the key "setting_label_key". 
                #The "True" return value (if used) will break the submenu
                #loop and allow to return to the main menu or
                #the nested menu.
                json_settings_dictionary = reset_to_default_setting("DPI Setting", json_settings_dictionary, 
                    json_default_settings_dictionary, json_settings_file_path_name)
                #A continue needs to be used, as we don't want 
                #the code below the "elif" statements to run,
                #which would cause a ValueError on int("r").
                continue
            elif choice == "q":
                quit_function()
            #The unit is removed (if provided)
            choice = re.sub(r"[ ]*dpi", "", choice)
            dpi_value = int(choice)
            if 50 <= dpi_value <= 600:
                #The "set_numeric_setting()" function will set the value of the setting found while accessing
                #the "json_settings_dictionary" dictionary with the key "setting_label_key" to the provided
                #value ("setting_value"). 
                json_settings_dictionary = set_numeric_setting(dpi_value, "DPI Setting", json_settings_dictionary, 
                json_settings_file_path_name)  
            else:
                input("\nInvalid choice, press any key to continue.")  
        except ValueError:
            input("\nInvalid choice, press any key to continue.")
    return json_settings_dictionary


#The "set_max_file_size()" will set the 'Maximal File Size' setting will set the size threshold, 
#in megabytes (MB), at which a new output PDF file will be generated (e.g., 'Book File (Part 2).pdf'). 
def set_max_file_size(json_settings_dictionary, json_default_settings_dictionary, json_settings_file_path_name):

    global is_in_submenu
    is_in_submenu = True

    while is_in_submenu:
        try:
            #The "clear_screen()" function will clear the CLI screen
            #using the appropriate command depending on the operating system.
            clear_screen()

            print("=== Set Max PDF File Size ===\n\n")

            #The function "get_terminal_dimensions()" will return the number of columns 
            #and rows in the console, to allow to properly format the text and dividers.
            columns, lines = get_terminal_dimensions()

            print(f"Current Setting: {json_settings_dictionary["Maximal File Size"]} MB | Default: {json_default_settings_dictionary["Maximal File Size"]} MB.\n")

            textwrapped_instructions_string = textwrap.fill(max_mb_per_pdf_file_comment_string, width=columns)
            textwrapped_input_string = textwrap.fill(f"Enter the max file size (5.0 MB or higher), or select one of the above options:", width=columns)     

            print(textwrapped_instructions_string)
            print(f"\n[r] Reset to the Default Setting\n[m] Main Menu\n[q] Quit\n")
            choice = input(textwrapped_input_string + " ").strip().lower()

            if choice == "":
                #A continue needs to be used, as we don't want 
                #the code below the "elif" statements to run,
                #which would cause a ValueError on int("").
                continue
            elif choice == "m":
                #The function "back_to_main_menu_function()"
                #will set the Boolean flags "is_in_submenu" and 
                #"is_in_sub_submenu" to "False", which will break the submenu
                #"while" loops and return to the main menu.
                back_to_main_menu_function(json_settings_dictionary)
                #A continue needs to be used, as we don't want 
                #the code below the "elif" statements to run,
                #which would cause a ValueError on int("m").
                continue
            elif choice == "r":
                #The "reset_to_default_setting()" function will reset the setting to its default value
                #found while accessing the value of the "json_default_settings_dictionary" dictionary 
                #with the key "setting_label_key". 
                #The "True" return value (if used) will break the submenu
                #loop and allow to return to the main menu or
                #the nested menu.
                json_settings_dictionary = reset_to_default_setting("Maximal File Size", json_settings_dictionary, 
                    json_default_settings_dictionary, json_settings_file_path_name)
                #A continue needs to be used, as we don't want 
                #the code below the "elif" statements to run,
                #which would cause a ValueError on int("r").
                continue
            elif choice == "q":
                quit_function()

            #The unit is removed (if provided)
            choice = re.sub(r"[ ]*mb", "", choice)

            max_pdf_file_size = float(choice)
            if max_pdf_file_size >= 5:
                #The "set_numeric_setting()" function will set the value of the setting found while accessing
                #the "json_settings_dictionary" dictionary with the key "setting_label_key" to the provided
                #value ("setting_value"). 
                json_settings_dictionary = set_numeric_setting(max_pdf_file_size, "Maximal File Size", json_settings_dictionary, 
                json_settings_file_path_name)
            else:
                input("\nInvalid choice, press any key to continue.") 
        except ValueError:
            input("\nInvalid choice, press any key to continue.")
    return json_settings_dictionary


#The "set_initial_brightness_level()" function will set the initial brightness level setting.
def set_initial_brightness_level(json_settings_dictionary, json_default_settings_dictionary, json_settings_file_path_name):

    global is_in_sub_submenu
    is_in_sub_submenu = True

    while is_in_sub_submenu:
        try:
            #The "clear_screen()" function will clear the CLI screen
            #using the appropriate command depending on the operating system.
            clear_screen()

            print("=== Set Initial Brightness Level ===\n\n")

            #The function "get_terminal_dimensions()" will return the number of columns 
            #and rows in the console, to allow to properly format the text and dividers.
            columns, lines = get_terminal_dimensions()

            print(f"Current Setting: {json_settings_dictionary["Initial Brightness Level"]} | Default: {json_default_settings_dictionary["Initial Brightness Level"]}.\n")

            textwrapped_instructions_string = textwrap.fill(initial_brightness_level_comment_string, width=columns)
            textwrapped_input_string = textwrap.fill(f"Enter the initial brightness level (greater than 0), or select one of the above options:", width=columns)

            print(textwrapped_instructions_string)
            print(f"\n[r] Reset to the Default Setting\n[b] Brightness Menu\n[m] Main Menu\n[q] Quit\n")
            choice = input(textwrapped_input_string + " ").strip().lower()

            if choice == "":
                #A continue needs to be used, as we don't want 
                #the code below the "elif" statements to run,
                #which would cause a ValueError on int("").
                continue
            elif choice == "m":
                #The function "back_to_main_menu_function()"
                #will set the Boolean flags "is_in_submenu" and 
                #"is_in_sub_submenu" to "False", which will break the submenu
                #"while" loops and return to the main menu.
                back_to_main_menu_function(json_settings_dictionary)
                #A continue needs to be used, as we don't want 
                #the code below the "elif" statements to run,
                #which would cause a ValueError on int("m").
                continue
            elif choice == "b":
                is_in_sub_submenu = False
                #A continue needs to be used, as we don't want 
                #the code below the "elif" statements to run,
                #which would cause a ValueError on int("b").
                continue
            elif choice == "r":
                #The "reset_to_default_setting()" function will reset the setting to its default value
                #found while accessing the value of the "json_default_settings_dictionary" dictionary 
                #with the key "setting_label_key". 
                #The "True" return value (if used) will break the submenu
                #loop and allow to return to the main menu or
                #the nested menu.
                json_settings_dictionary = reset_to_default_setting("Initial Brightness Level", json_settings_dictionary, 
                    json_default_settings_dictionary, json_settings_file_path_name)
                #A continue needs to be used, as we don't want 
                #the code below the "elif" statements to run,
                #which would cause a ValueError on int("r").
                continue
            elif choice == "q":
                quit_function()
            initial_brightness_level = float(choice)
            if initial_brightness_level > 0:
                #The "set_numeric_setting()" function will set the value of the setting found while accessing
                #the "json_settings_dictionary" dictionary with the key "setting_label_key" to the provided
                #value ("setting_value"). 
                json_settings_dictionary = set_numeric_setting(initial_brightness_level, "Initial Brightness Level", json_settings_dictionary, 
                json_settings_file_path_name) 
            else:
                input("\nInvalid choice, press any key to continue.")  
        except ValueError:
            input("\nInvalid choice, press any key to continue.")
    return json_settings_dictionary


#The "set_final_brightness_level()" function will set the final brightness level setting.
def set_final_brightness_level(json_settings_dictionary, json_default_settings_dictionary, json_settings_file_path_name):

    global is_in_sub_submenu
    is_in_sub_submenu = True

    while is_in_sub_submenu:
        try:
            #The "clear_screen()" function will clear the CLI screen
            #using the appropriate command depending on the operating system.
            clear_screen()

            print("=== Set Final Brightness Level ===\n\n")

            #The function "get_terminal_dimensions()" will return the number of columns 
            #and rows in the console, to allow to properly format the text and dividers.
            columns, lines = get_terminal_dimensions()

            print(f"Current Setting: {json_settings_dictionary["Final Brightness Level"]} | Default: {json_default_settings_dictionary["Final Brightness Level"]}.\n")

            textwrapped_instructions_string = textwrap.fill(final_brightness_level_comment_string, width=columns)
            textwrapped_input_string = textwrap.fill(f"Enter the final brightness level (greater than 0), or select one of the above options:", width=columns)

            print(textwrapped_instructions_string)
            print(f"\n[r] Reset to the Default Setting\n[b] Brightness Menu\n[m] Main Menu\n[q] Quit\n")
            choice = input(textwrapped_input_string + " ").strip().lower()

            if choice == "":
                #A continue needs to be used, as we don't want 
                #the code below the "elif" statements to run,
                #which would cause a ValueError on int("").
                continue
            elif choice == "m":
                #The function "back_to_main_menu_function()"
                #will set the Boolean flags "is_in_submenu" and 
                #"is_in_sub_submenu" to "False", which will break the submenu
                #"while" loops and return to the main menu.
                back_to_main_menu_function(json_settings_dictionary)
                #A continue needs to be used, as we don't want 
                #the code below the "elif" statements to run,
                #which would cause a ValueError on int("m").
                continue
            elif choice == "b":
                is_in_sub_submenu = False
                #A continue needs to be used, as we don't want 
                #the code below the "elif" statements to run,
                #which would cause a ValueError on int("b").
                continue
            elif choice == "r":
                #The "reset_to_default_setting()" function will reset the setting to its default value
                #found while accessing the value of the "json_default_settings_dictionary" dictionary 
                #with the key "setting_label_key". 
                #The "True" return value (if used) will break the submenu
                #loop and allow to return to the main menu or
                #the nested menu.
                json_settings_dictionary = reset_to_default_setting("Final Brightness Level", json_settings_dictionary, 
                    json_default_settings_dictionary, json_settings_file_path_name)
                #A continue needs to be used, as we don't want 
                #the code below the "elif" statements to run,
                #which would cause a ValueError on int("r").
                continue
            elif choice == "q":
                quit_function()
            final_brightness_level = float(choice)
            if final_brightness_level > 0:
                #The "set_numeric_setting()" function will set the value of the setting found while accessing
                #the "json_settings_dictionary" dictionary with the key "setting_label_key" to the provided
                #value ("setting_value"). 
                json_settings_dictionary = set_numeric_setting(final_brightness_level, "Final Brightness Level", json_settings_dictionary, 
                json_settings_file_path_name)
            else:
                input("\nInvalid choice, press any key to continue.")  
        except ValueError:
            input("\nInvalid choice, press any key to continue.")
    return json_settings_dictionary


#The "brightness_levels_menu()" function will run a "while is_in_submenu"
#loop that will allow the user to navigate the menu, and the loop will 
#be broken out of when they select the "Quit" option.
def brightness_levels_menu(json_settings_dictionary, json_default_settings_dictionary, json_settings_file_path_name):

    global is_in_submenu
    is_in_submenu = True

    while is_in_submenu:
        #The "clear_screen()" function will clear the CLI screen
        #using the appropriate command depending on the operating system.
        clear_screen()

        brightness_levels_menu_actions_dict = {
        "1": ["Set Initial Brightness Level (Optional)", set_initial_brightness_level, (json_settings_dictionary, json_default_settings_dictionary, json_settings_file_path_name)],
        "2": ["Set Final Brightness Level (Optional)", set_final_brightness_level, (json_settings_dictionary, json_default_settings_dictionary, json_settings_file_path_name)],
        "m": ["Main Menu", back_to_main_menu_function, (json_settings_dictionary,)],
        "q": ["Quit", quit_function, ()]}

        #The function "textwrap_action_strings_in_menu_action_dict()", which takes in 
        #a menu action dictionary comprised of one character keys and values made up
        #of a three-member tuple (action string, function, function arguments).
        #The action strings ("value[0]") will be textwrapped and the modified
        #dictionary will be returned.
        brightness_levels_menu_actions_dict = textwrap_action_strings_in_menu_action_dict(brightness_levels_menu_actions_dict)

        print("=== Brightness Menu ===\n\n")

        #The function "run_menu" will retrieve and call the function
        #at the appropriate choice key in the "menu_actions_dict"
        json_settings_dictionary = run_menu(brightness_levels_menu_actions_dict, json_settings_dictionary)
    return json_settings_dictionary


#The "set_initial_contrast_level()" function will set the initial contrast level setting.
def set_initial_contrast_level(json_settings_dictionary, json_default_settings_dictionary, json_settings_file_path_name):

    global is_in_sub_submenu
    is_in_sub_submenu = True

    while is_in_sub_submenu:
        try:
            #The "clear_screen()" function will clear the CLI screen
            #using the appropriate command depending on the operating system.
            clear_screen()

            print("=== Set Initial Contrast Level ===\n\n")

            #The function "get_terminal_dimensions()" will return the number of columns 
            #and rows in the console, to allow to properly format the text and dividers.
            columns, lines = get_terminal_dimensions()

            print(f"Current Setting: {json_settings_dictionary["Initial Contrast Level"]} | Default: {json_default_settings_dictionary["Initial Contrast Level"]}.\n")

            textwrapped_instructions_string = textwrap.fill(initial_contrast_level_comment_string, width=columns)
            textwrapped_input_string = textwrap.fill(f"Enter the initial contrast level (0 or higher), or select one of the above options:", width=columns)

            print(textwrapped_instructions_string)
            print(f"\n[r] Reset to the Default Setting\n[b] Contrast Menu\n[m] Main Menu\n[q] Quit\n")
            choice = input(textwrapped_input_string + " ").strip().lower()

            if choice == "":
                #A continue needs to be used, as we don't want 
                #the code below the "elif" statements to run,
                #which would cause a ValueError on int("").
                continue
            elif choice == "m":
                #The function "back_to_main_menu_function()"
                #will set the Boolean flags "is_in_submenu" and 
                #"is_in_sub_submenu" to "False", which will break the submenu
                #"while" loops and return to the main menu.
                back_to_main_menu_function(json_settings_dictionary)
                #A continue needs to be used, as we don't want 
                #the code below the "elif" statements to run,
                #which would cause a ValueError on int("m").
                continue
            elif choice == "b":
                is_in_sub_submenu = False
                #A continue needs to be used, as we don't want 
                #the code below the "elif" statements to run,
                #which would cause a ValueError on int("b").
                continue
            elif choice == "r":
                #The "reset_to_default_setting()" function will reset the setting to its default value
                #found while accessing the value of the "json_default_settings_dictionary" dictionary 
                #with the key "setting_label_key". 
                #The "True" return value (if used) will break the submenu
                #loop and allow to return to the main menu or
                #the nested menu.
                json_settings_dictionary = reset_to_default_setting("Initial Contrast Level", json_settings_dictionary, 
                    json_default_settings_dictionary, json_settings_file_path_name)
                #A continue needs to be used, as we don't want 
                #the code below the "elif" statements to run,
                #which would cause a ValueError on int("r").
                continue
            elif choice == "q":
                quit_function()
            initial_contrast_level = float(choice)
            if initial_contrast_level >= 0:
                #The "set_numeric_setting()" function will set the value of the setting found while accessing
                #the "json_settings_dictionary" dictionary with the key "setting_label_key" to the provided
                #value ("setting_value"). 
                json_settings_dictionary = set_numeric_setting(initial_contrast_level, "Initial Contrast Level", json_settings_dictionary, 
                json_settings_file_path_name)
            else:
                input("\nInvalid choice, press any key to continue.")  
        except ValueError:
            input("\nInvalid choice, press any key to continue.")
    return json_settings_dictionary


#The "set_final_contrast_level()" function will set the final contrast level setting.
def set_final_contrast_level(json_settings_dictionary, json_default_settings_dictionary, json_settings_file_path_name):

    global is_in_sub_submenu
    is_in_sub_submenu = True

    while is_in_sub_submenu:
        try:
            #The "clear_screen()" function will clear the CLI screen
            #using the appropriate command depending on the operating system.
            clear_screen()

            print("=== Set Final Contrast Level ===\n\n")

            #The function "get_terminal_dimensions()" will return the number of columns 
            #and rows in the console, to allow to properly format the text and dividers.
            columns, lines = get_terminal_dimensions()

            print(f"Current Setting: {json_settings_dictionary["Final Contrast Level"]} | Default: {json_default_settings_dictionary["Final Contrast Level"]}.\n")

            textwrapped_instructions_string = textwrap.fill(final_contrast_level_comment_string, width=columns)
            textwrapped_input_string = textwrap.fill(f"Enter the final contrast level (0 or higher), or select one of the above options:", width=columns)

            print(textwrapped_instructions_string)
            print(f"\n[r] Reset to the Default Setting\n[b] Contrast Menu\n[m] Main Menu\n[q] Quit\n")
            choice = input(textwrapped_input_string + " ").strip().lower()

            if choice == "":
                #A continue needs to be used, as we don't want 
                #the code below the "elif" statements to run,
                #which would cause a ValueError on int("").
                continue
            elif choice == "m":
                #The function "back_to_main_menu_function()"
                #will set the Boolean flags "is_in_submenu" and 
                #"is_in_sub_submenu" to "False", which will break the submenu
                #"while" loops and return to the main menu.
                back_to_main_menu_function(json_settings_dictionary)
                #A continue needs to be used, as we don't want 
                #the code below the "elif" statements to run,
                #which would cause a ValueError on int("m").
                continue
            elif choice == "b":
                is_in_sub_submenu = False
                #A continue needs to be used, as we don't want 
                #the code below the "elif" statements to run,
                #which would cause a ValueError on int("b").
                continue
            elif choice == "r":
                #The "reset_to_default_setting()" function will reset the setting to its default value
                #found while accessing the value of the "json_default_settings_dictionary" dictionary 
                #with the key "setting_label_key". 
                #The "True" return value (if used) will break the submenu
                #loop and allow to return to the main menu or
                #the nested menu.
                json_settings_dictionary = reset_to_default_setting("Final Contrast Level", json_settings_dictionary, 
                    json_default_settings_dictionary, json_settings_file_path_name)
                #A continue needs to be used, as we don't want 
                #the code below the "elif" statements to run,
                #which would cause a ValueError on int("r").
                continue
            elif choice == "q":
                quit_function()
            final_contrast_level = float(choice)
            if final_contrast_level >= 0:
                #The "set_numeric_setting()" function will set the value of the setting found while accessing
                #the "json_settings_dictionary" dictionary with the key "setting_label_key" to the provided
                #value ("setting_value"). 
                json_settings_dictionary = set_numeric_setting(final_contrast_level, "Final Contrast Level", json_settings_dictionary, 
                json_settings_file_path_name)
            else:
                input("\nInvalid choice, press any key to continue.")  
        except ValueError:
            input("\nInvalid choice, press any key to continue.")
    return json_settings_dictionary


#The "contrast_levels_menu()" function will run a "while is_in_submenu"
#loop that will allow the user to navigate the menu, and the loop will 
#be broken out of when they select the "Quit" option.
def contrast_levels_menu(json_settings_dictionary, json_default_settings_dictionary, json_settings_file_path_name):

    global is_in_submenu
    is_in_submenu = True

    while is_in_submenu:
        #The "clear_screen()" function will clear the CLI screen
        #using the appropriate command depending on the operating system.
        clear_screen()

        contrast_levels_menu_actions_dict = {
        "1": ["Set Initial Contrast Level (Optional)", set_initial_contrast_level, (json_settings_dictionary, json_default_settings_dictionary, json_settings_file_path_name)],
        "2": ["Set Final Contrast Level (Optional)", set_final_contrast_level, (json_settings_dictionary, json_default_settings_dictionary, json_settings_file_path_name)],
        "m": ["Main Menu", back_to_main_menu_function, (json_settings_dictionary,)],
        "q": ["Quit", quit_function, ()]}

        #The function "textwrap_action_strings_in_menu_action_dict()", which takes in 
        #a menu action dictionary comprised of one character keys and values made up
        #of a three-member tuple (action string, function, function arguments).
        #The action strings ("value[0]") will be textwrapped and the modified
        #dictionary will be returned.
        contrast_levels_menu_actions_dict = textwrap_action_strings_in_menu_action_dict(contrast_levels_menu_actions_dict)

        print("=== Contrast Menu ===\n\n")

        #The function "run_menu" will retrieve and call the function
        #at the appropriate choice key in the "menu_actions_dict"
        json_settings_dictionary = run_menu(contrast_levels_menu_actions_dict, json_settings_dictionary)
    return json_settings_dictionary


#The "set_initial_page_color_filter_multiplier()" function will set the modifier for the page color filter.
def set_initial_page_color_filter_multiplier(json_settings_dictionary, json_default_settings_dictionary, json_settings_file_path_name):

    global is_in_sub_sub_submenu
    is_in_sub_sub_submenu = True

    while is_in_sub_sub_submenu:
        try:
            #The "clear_screen()" function will clear the CLI screen
            #using the appropriate command depending on the operating system.
            clear_screen()

            print("=== Set Initial Page Color Filter Multiplier ===\n\n")

            #The function "get_terminal_dimensions()" will return the number of columns 
            #and rows in the console, to allow to properly format the text and dividers.
            columns, lines = get_terminal_dimensions()

            print(f"Current Setting: {json_settings_dictionary["Page Color Filter Multiplier"]} | Default: {json_default_settings_dictionary["Page Color Filter Multiplier"]}.\n")

            textwrapped_instructions_string = textwrap.fill(number_of_standard_deviations_for_filtering_page_color_comment_string, width=columns)
            textwrapped_input_string = textwrap.fill(f"Enter the value of the multiplier (-3.00 to +3.00), or select one of the above options:", width=columns)

            print(textwrapped_instructions_string)
            print(f"\n[r] Reset to the Default Setting\n[b] Page Color Filter Menu\n[m] Main Menu\n[q] Quit\n")
            choice = input(textwrapped_input_string + " ").strip().lower()

            if choice == "":
                #A continue needs to be used, as we don't want 
                #the code below the "elif" statements to run,
                #which would cause a ValueError on int("").
                continue
            elif choice == "m":
                #The function "back_to_main_menu_function()"
                #will set the Boolean flags "is_in_submenu" and 
                #"is_in_sub_submenu" to "False", which will break the submenu
                #"while" loops and return to the main menu.
                back_to_main_menu_function(json_settings_dictionary)
                #A continue needs to be used, as we don't want 
                #the code below the "elif" statements to run,
                #which would cause a ValueError on int("m").
                continue
            elif choice == "b":
                is_in_sub_sub_submenu = False
                #A continue needs to be used, as we don't want 
                #the code below the "elif" statements to run,
                #which would cause a ValueError on int("b").
                continue
            elif choice == "r":
                #The "reset_to_default_setting()" function will reset the setting to its default value
                #found while accessing the value of the "json_default_settings_dictionary" dictionary 
                #with the key "setting_label_key". 
                #The "True" return value (if used) will break the submenu
                #loop and allow to return to the main menu or
                #the nested menu.
                json_settings_dictionary = reset_to_default_setting("Page Color Filter Multiplier", json_settings_dictionary, 
                    json_default_settings_dictionary, json_settings_file_path_name)
                #A continue needs to be used, as we don't want 
                #the code below the "elif" statements to run,
                #which would cause a ValueError on int("r").
                continue
            elif choice == "q":
                quit_function()
            page_color_filter_multiplier = float(choice)
            if -3.0 <= page_color_filter_multiplier <= 3.0:
                #The "set_numeric_setting()" function will set the value of the setting found while accessing
                #the "json_settings_dictionary" dictionary with the key "setting_label_key" to the provided
                #value ("setting_value"). 
                json_settings_dictionary = set_numeric_setting(page_color_filter_multiplier, "Page Color Filter Multiplier", json_settings_dictionary, 
                json_settings_file_path_name)
            else:
                input("\nInvalid choice, press any key to continue.") 
        except ValueError:
            input("\nInvalid choice, press any key to continue.")
    return json_settings_dictionary


#The "set_initial_page_color_filter_multiplier_when_cropping()" function will set the modifier for the page color filter when cropping.
#This filter's multiplier is usually lower (more aggressively filtering) than the regular "initial_page_color_filter", as no blemishes
#should remain on the page to ensure that it gets cropped nicely.
def set_initial_page_color_filter_multiplier_when_cropping(json_settings_dictionary, json_default_settings_dictionary, json_settings_file_path_name):

    global is_in_sub_sub_submenu
    is_in_sub_sub_submenu = True

    while is_in_sub_sub_submenu:
        try:
            #The "clear_screen()" function will clear the CLI screen
            #using the appropriate command depending on the operating system.
            clear_screen()

            print("=== Set Initial Page Color Filter Multiplier When Cropping ===\n\n")

            #The function "get_terminal_dimensions()" will return the number of columns 
            #and rows in the console, to allow to properly format the text and dividers.
            columns, lines = get_terminal_dimensions()

            print(f"Current Setting: {json_settings_dictionary["Page Color Filter Multiplier When Cropping"]} | Default: {json_default_settings_dictionary["Page Color Filter Multiplier When Cropping"]}.\n")

            textwrapped_instructions_string = textwrap.fill(number_of_standard_deviations_for_filtering_page_color_when_cropping_comment_string, width=columns)
            textwrapped_input_string = textwrap.fill(f"Enter the value of the multiplier (-3.00 to +3.00), or select one of the above options:", width=columns)

            print(textwrapped_instructions_string)
            print(f"\n[r] Reset to the Default Setting\n[b] Page Color Filter Menu\n[m] Main Menu\n[q] Quit\n")
            choice = input(textwrapped_input_string + " ").strip().lower()

            if choice == "":
                #A continue needs to be used, as we don't want 
                #the code below the "elif" statements to run,
                #which would cause a ValueError on int("").
                continue
            elif choice == "m":
                #The function "back_to_main_menu_function()"
                #will set the Boolean flags "is_in_submenu" and 
                #"is_in_sub_submenu" to "False", which will break the submenu
                #"while" loops and return to the main menu.
                back_to_main_menu_function(json_settings_dictionary)
                #A continue needs to be used, as we don't want 
                #the code below the "elif" statements to run,
                #which would cause a ValueError on int("m").
                continue
            elif choice == "b":
                is_in_sub_sub_submenu = False
                #A continue needs to be used, as we don't want 
                #the code below the "elif" statements to run,
                #which would cause a ValueError on int("b").
                continue
            elif choice == "r":
                #The "reset_to_default_setting()" function will reset the setting to its default value
                #found while accessing the value of the "json_default_settings_dictionary" dictionary 
                #with the key "setting_label_key". 
                #The "True" return value (if used) will break the submenu
                #loop and allow to return to the main menu or
                #the nested menu.
                json_settings_dictionary = reset_to_default_setting("Page Color Filter Multiplier When Cropping", json_settings_dictionary, 
                    json_default_settings_dictionary, json_settings_file_path_name)
                #A continue needs to be used, as we don't want 
                #the code below the "elif" statements to run,
                #which would cause a ValueError on int("r").
                continue
            elif choice == "q":
                quit_function()
            page_color_filter_multiplier = float(choice)
            if -3.0 <= page_color_filter_multiplier <= 3.0:
                #The "set_numeric_setting()" function will set the value of the setting found while accessing
                #the "json_settings_dictionary" dictionary with the key "setting_label_key" to the provided
                #value ("setting_value"). 
                json_settings_dictionary = set_numeric_setting(page_color_filter_multiplier, "Page Color Filter Multiplier When Cropping", json_settings_dictionary, 
                json_settings_file_path_name)
            else:
                input("\nInvalid choice, press any key to continue.") 
        except ValueError:
            input("\nInvalid choice, press any key to continue.")
    return json_settings_dictionary


#The "page_color_filter_menu()" function will run a "while is_in_submenu"
#loop that will allow the user to navigate the menu, and the loop will 
#be broken out of when they select the "Quit" option.
def page_color_filter_menu(json_settings_dictionary, json_default_settings_dictionary, json_settings_file_path_name):

    global is_in_sub_submenu
    is_in_sub_submenu = True

    while is_in_sub_submenu:
        #The "clear_screen()" function will clear the CLI screen
        #using the appropriate command depending on the operating system.
        clear_screen()

        page_color_filter_menu_actions_dict = {   
        "1": ["Set Page Color Filter Multiplier (Removes the page color)", set_initial_page_color_filter_multiplier, (json_settings_dictionary, json_default_settings_dictionary, json_settings_file_path_name)],
        "2": ["Set Page Color Filter Multiplier When Cropping (Helps to properly crop the pages)", set_initial_page_color_filter_multiplier_when_cropping, (json_settings_dictionary, json_default_settings_dictionary, json_settings_file_path_name)],
        "b": ["Filters Menu", back_to_submenu_function, (json_settings_dictionary,)],
        "m": ["Main Menu", back_to_main_menu_function, (json_settings_dictionary,)],
        "q": ["Quit", quit_function, ()]}

        #The function "textwrap_action_strings_in_menu_action_dict()", which takes in 
        #a menu action dictionary comprised of one character keys and values made up
        #of a three-member tuple (action string, function, function arguments).
        #The action strings ("value[0]") will be textwrapped and the modified
        #dictionary will be returned.
        page_color_filter_menu_actions_dict = textwrap_action_strings_in_menu_action_dict(page_color_filter_menu_actions_dict)

        print("=== Page Color Filter Menu ===\n\n")

        #The function "run_menu" will retrieve and call the function
        #at the appropriate choice key in the "menu_actions_dict"
        json_settings_dictionary = run_menu(page_color_filter_menu_actions_dict, json_settings_dictionary)
    return json_settings_dictionary


#The "set_margins_filter_multiplier()" function will set the modifier for the page color filter.
def set_margins_filter_multiplier(json_settings_dictionary, json_default_settings_dictionary, json_settings_file_path_name):

    global is_in_sub_sub_submenu
    is_in_sub_sub_submenu = True

    while is_in_sub_sub_submenu:
        try:
            #The "clear_screen()" function will clear the CLI screen
            #using the appropriate command depending on the operating system.
            clear_screen()

            print("=== Set Margins Filter Multiplier ===\n\n")

            #The function "get_terminal_dimensions()" will return the number of columns 
            #and rows in the console, to allow to properly format the text and dividers.
            columns, lines = get_terminal_dimensions()

            textwrapped_instructions_string = textwrap.fill(number_of_standard_deviations_for_filtering_splotches_margins_comment_string, width=columns)
            textwrapped_input_string = textwrap.fill(f"Enter the value of the multiplier (-3.00 to +3.00), or select one of the above options:", width=columns)

            if (json_settings_dictionary["Margins Filter"]):
                print("Margins filter is currently turned ON (Default value).\n")
            else:
                print("Margins filter is currently turned OFF.\n")

            print(f"Current Setting: {json_settings_dictionary["Margins Filter Multiplier"]} | Default: {json_default_settings_dictionary["Margins Filter Multiplier"]}.\n")

            print(textwrapped_instructions_string)
            print(f"\n[t] Toggle Filter On/Off\n[r] Reset to the Default Setting\n[b] Margins Filter Menu\n[m] Main Menu\n[q] Quit\n")

            choice = input(textwrapped_input_string + " ").strip().lower()

            if choice == "":
                #A continue needs to be used, as we don't want 
                #the code below the "elif" statements to run,
                #which would cause a ValueError on int("").
                continue
            elif choice == "m":
                #The function "back_to_main_menu_function()"
                #will set the Boolean flags "is_in_submenu" and 
                #"is_in_sub_submenu" to "False", which will break the submenu
                #"while" loops and return to the main menu.
                back_to_main_menu_function(json_settings_dictionary)
                #A continue needs to be used, as we don't want 
                #the code below the "elif" statements to run,
                #which would cause a ValueError on int("m").
                continue
            elif choice == "b":
                is_in_sub_sub_submenu = False
                #A continue needs to be used, as we don't want 
                #the code below the "elif" statements to run,
                #which would cause a ValueError on int("b").
                continue
            elif choice == "r":
                #The "reset_to_default_setting()" function will reset the setting to its default value
                #found while accessing the value of the "json_default_settings_dictionary" dictionary 
                #with the key "setting_label_key". 
                #The "True" return value (if used) will break the submenu
                #loop and allow to return to the main menu or
                #the nested menu.
                json_settings_dictionary = reset_to_default_setting("Margins Filter Multiplier", json_settings_dictionary, 
                    json_default_settings_dictionary, json_settings_file_path_name)
                #A continue needs to be used, as we don't want 
                #the code below the "elif" statements to run,
                #which would cause a ValueError on int("r").
                continue
            elif choice == "t":
                #The "toggle_boolean_setting()" function will set the Boolean setting found while accessing
                #the "json_settings_dictionary" dictionary with the key "setting_label_key" to the opposite
                #value of the current setting ("True" if the current setting is "False" and vice-versa). 

                #The outcome string will be returned from the "toggle_boolean_setting()" 
                #function all if the default parameter Boolean flag "do_not_print_results" 
                #is set to "True", to ensure that the same state of the Filter is printed 
                #both at the top of the screen (current state) and in the confirmation string.
                json_settings_dictionary = toggle_boolean_setting("Margins Filter", json_settings_dictionary, 
                    json_settings_file_path_name)
                #A continue needs to be used, as we don't want 
                #the code below the "elif" statements to run,
                #which would cause a ValueError on int("r").
                continue
            elif choice == "q":
                quit_function()

            margins_filter_multiplier = float(choice)
            if -3.0 <= margins_filter_multiplier <= 3.0:
                #The "set_numeric_setting()" function will set the value of the setting found while accessing
                #the "json_settings_dictionary" dictionary with the key "setting_label_key" to the provided
                #value ("setting_value"). 
                json_settings_dictionary = set_numeric_setting(margins_filter_multiplier, "Margins Filter Multiplier", json_settings_dictionary, 
                    json_settings_file_path_name)
            else:
                input("\nInvalid choice, press any key to continue.")   
        except ValueError:
            input("\nInvalid choice, press any key to continue.")
    return json_settings_dictionary


#The "set_margins_filter_left_margin()" function will set the left margin for the margins filter.
#It will also allow the user to toggle the "Margins Filter" on or off.
def set_margins_filter_left_margin(json_settings_dictionary, json_default_settings_dictionary, json_settings_file_path_name):

    global is_in_sub_sub_submenu
    is_in_sub_sub_submenu = True

    while is_in_sub_sub_submenu:
        try:
            #The "clear_screen()" function will clear the CLI screen
            #using the appropriate command depending on the operating system.
            clear_screen()

            print("=== Set Margins Filter Left Margin ===\n\n")

            #The function "get_terminal_dimensions()" will return the number of columns 
            #and rows in the console, to allow to properly format the text and dividers.
            columns, lines = get_terminal_dimensions()

            textwrapped_instructions_string = textwrap.fill(left_margin_width_percent_comment_string, width=columns)
            textwrapped_input_string = textwrap.fill(f"Enter the left margin (0% or higher), or select one of the above options:", width=columns)

            if (json_settings_dictionary["Margins Filter"]):
                print("Margins filter is currently turned ON (Default value).\n")
            else:
                print("Margins filter is currently turned OFF.\n")

            print(f"Current Setting: {json_settings_dictionary["Margins Filter Left Margin"]}% | Default: {json_default_settings_dictionary["Margins Filter Left Margin"]}%.\n")

            print(textwrapped_instructions_string)
            print(f"\n[t] Toggle Filter On/Off\n[r] Reset to the Default Setting\n[b] Margins Filter Menu\n[m] Main Menu\n[q] Quit\n")  

            choice = input(textwrapped_input_string + " ").strip().lower()

            if choice == "":
                #A continue needs to be used, as we don't want 
                #the code below the "elif" statements to run,
                #which would cause a ValueError on int("").
                continue
            elif choice == "m":
                #The function "back_to_main_menu_function()"
                #will set the Boolean flags "is_in_submenu" and 
                #"is_in_sub_submenu" to "False", which will break the submenu
                #"while" loops and return to the main menu.
                back_to_main_menu_function(json_settings_dictionary)
                #A continue needs to be used, as we don't want 
                #the code below the "elif" statements to run,
                #which would cause a ValueError on int("m").
                continue
            elif choice == "b":
                is_in_sub_sub_submenu = False
                #A continue needs to be used, as we don't want 
                #the code below the "elif" statements to run,
                #which would cause a ValueError on int("b").
                continue
            elif choice == "r":
                #The "reset_to_default_setting()" function will reset the setting to its default value
                #found while accessing the value of the "json_default_settings_dictionary" dictionary 
                #with the key "setting_label_key". 
                #The "True" return value (if used) will break the submenu
                #loop and allow to return to the main menu or
                #the nested menu.
                json_settings_dictionary = reset_to_default_setting("Margins Filter Left Margin", json_settings_dictionary, 
                    json_default_settings_dictionary, json_settings_file_path_name)
                #A continue needs to be used, as we don't want 
                #the code below the "elif" statements to run,
                #which would cause a ValueError on int("r").
                continue
            elif choice == "t":
                #The "toggle_boolean_setting()" function will set the Boolean setting found while accessing
                #the "json_settings_dictionary" dictionary with the key "setting_label_key" to the opposite
                #value of the current setting ("True" if the current setting is "False" and vice-versa). 

                #The outcome string will be returned from the "toggle_boolean_setting()" 
                #function all if the default parameter Boolean flag "do_not_print_results" 
                #is set to "True", to ensure that the same state of the Filter is printed 
                #both at the top of the screen (current state) and in the confirmation string.
                json_settings_dictionary = toggle_boolean_setting("Margins Filter", json_settings_dictionary, 
                    json_settings_file_path_name)
                #A continue needs to be used, as we don't want 
                #the code below the "elif" statements to run,
                #which would cause a ValueError on int("r").
                continue
            elif choice == "q":
                quit_function()

            #The unit is removed (if provided)
            choice = re.sub(r"[ ]*%", "", choice)

            margin_width_percent = float(choice)
            if margin_width_percent >= 0:
                #The "set_numeric_setting()" function will set the value of the setting found while accessing
                #the "json_settings_dictionary" dictionary with the key "setting_label_key" to the provided
                #value ("setting_value"). 
                json_settings_dictionary = set_numeric_setting(margin_width_percent, "Margins Filter Left Margin", json_settings_dictionary, 
                    json_settings_file_path_name)
            else:
                input("\nInvalid choice, press any key to continue.")  
        except ValueError:
            input("\nInvalid choice, press any key to continue.")
    return json_settings_dictionary


#The "set_margins_filter_right_margin()" function will set the right margin for the margins filter.
#It will also allow the user to toggle the "Margins Filter" on or off.
def set_margins_filter_right_margin(json_settings_dictionary, json_default_settings_dictionary, json_settings_file_path_name):

    global is_in_sub_sub_submenu
    is_in_sub_sub_submenu = True

    while is_in_sub_sub_submenu:
        try:
            #The "clear_screen()" function will clear the CLI screen
            #using the appropriate command depending on the operating system.
            clear_screen()

            print("=== Set Margins Filter Right Margin ===\n\n")

            #The function "get_terminal_dimensions()" will return the number of columns 
            #and rows in the console, to allow to properly format the text and dividers.
            columns, lines = get_terminal_dimensions()

            textwrapped_instructions_string = textwrap.fill(right_margin_width_percent_comment_string, width=columns)
            textwrapped_input_string = textwrap.fill(f"Enter the right margin (0% or higher), or select one of the above options:", width=columns)

            if (json_settings_dictionary["Margins Filter"]):
                print("Margins filter is currently turned ON (Default value).\n")
            else:
                print("Margins filter is currently turned OFF.\n")

            print(f"Current Setting: {json_settings_dictionary["Margins Filter Right Margin"]}% | Default: {json_default_settings_dictionary["Margins Filter Right Margin"]}%.\n")

            print(textwrapped_instructions_string)
            print(f"\n[t] Toggle Filter On/Off\n[r] Reset to the Default Setting\n[b] Margins Filter Menu\n[m] Main Menu\n[q] Quit\n")

            choice = input(textwrapped_input_string + " ").strip().lower()

            if choice == "":
                #A continue needs to be used, as we don't want 
                #the code below the "elif" statements to run,
                #which would cause a ValueError on int("").
                continue
            elif choice == "m":
                #The function "back_to_main_menu_function()"
                #will set the Boolean flags "is_in_submenu" and 
                #"is_in_sub_submenu" to "False", which will break the submenu
                #"while" loops and return to the main menu.
                back_to_main_menu_function(json_settings_dictionary)
                #A continue needs to be used, as we don't want 
                #the code below the "elif" statements to run,
                #which would cause a ValueError on int("m").
                continue
            elif choice == "b":
                is_in_sub_sub_submenu = False
                #A continue needs to be used, as we don't want 
                #the code below the "elif" statements to run,
                #which would cause a ValueError on int("b").
                continue
            elif choice == "r":
                #The "reset_to_default_setting()" function will reset the setting to its default value
                #found while accessing the value of the "json_default_settings_dictionary" dictionary 
                #with the key "setting_label_key". 
                #The "True" return value (if used) will break the submenu
                #loop and allow to return to the main menu or
                #the nested menu.
                json_settings_dictionary = reset_to_default_setting("Margins Filter Right Margin", json_settings_dictionary, 
                    json_default_settings_dictionary, json_settings_file_path_name)
                #A continue needs to be used, as we don't want 
                #the code below the "elif" statements to run,
                #which would cause a ValueError on int("r").
                continue
            elif choice == "t":
                #The "toggle_boolean_setting()" function will set the Boolean setting found while accessing
                #the "json_settings_dictionary" dictionary with the key "setting_label_key" to the opposite
                #value of the current setting ("True" if the current setting is "False" and vice-versa). 

                #The outcome string will be returned from the "toggle_boolean_setting()" 
                #function all if the default parameter Boolean flag "do_not_print_results" 
                #is set to "True", to ensure that the same state of the Filter is printed 
                #both at the top of the screen (current state) and in the confirmation string.
                json_settings_dictionary = toggle_boolean_setting("Margins Filter", json_settings_dictionary, 
                    json_settings_file_path_name)
                #A continue needs to be used, as we don't want 
                #the code below the "elif" statements to run,
                #which would cause a ValueError on int("r").
                continue
            elif choice == "q":
                quit_function()

            #The unit is removed (if provided)
            choice = re.sub(r"[ ]*%", "", choice)

            margin_width_percent = float(choice)
            if margin_width_percent >= 0:
                #The "set_numeric_setting()" function will set the value of the setting found while accessing
                #the "json_settings_dictionary" dictionary with the key "setting_label_key" to the provided
                #value ("setting_value"). 
                json_settings_dictionary = set_numeric_setting(margin_width_percent, "Margins Filter Right Margin", json_settings_dictionary, 
                    json_settings_file_path_name)
            else:
                input("\nInvalid choice, press any key to continue.")  
        except ValueError:
            input("\nInvalid choice, press any key to continue.")
    return json_settings_dictionary


#The "set_margins_filter_top_margin()" function will set the top margin for the margins filter.
#It will also allow the user to toggle the "Margins Filter" on or off.
def set_margins_filter_top_margin(json_settings_dictionary, json_default_settings_dictionary, json_settings_file_path_name):

    global is_in_sub_sub_submenu
    is_in_sub_sub_submenu = True

    while is_in_sub_sub_submenu:
        try:
            #The "clear_screen()" function will clear the CLI screen
            #using the appropriate command depending on the operating system.
            clear_screen()

            print("=== Set Margins Filter Top Margin ===\n\n")

            #The function "get_terminal_dimensions()" will return the number of columns 
            #and rows in the console, to allow to properly format the text and dividers.
            columns, lines = get_terminal_dimensions()

            textwrapped_instructions_string = textwrap.fill(top_margin_height_percent_comment_string, width=columns)
            textwrapped_input_string = textwrap.fill(f"Enter the top margin setting (0% or higher), or select one of the above options:", width=columns)

            if (json_settings_dictionary["Margins Filter"]):
                print("Margins filter is currently turned ON (Default value).\n")
            else:
                print("Margins filter is currently turned OFF.\n")

            print(f"Current Setting: {json_settings_dictionary["Margins Filter Top Margin"]}% | Default: {json_default_settings_dictionary["Margins Filter Top Margin"]}%.\n")

            print(textwrapped_instructions_string)
            print(f"\n[t] Toggle Filter On/Off\n[r] Reset to the Default Setting\n[b] Margins Filter Menu\n[m] Main Menu\n[q] Quit\n")

            choice = input(textwrapped_input_string + " ").strip().lower()

            if choice == "":
                #A continue needs to be used, as we don't want 
                #the code below the "elif" statements to run,
                #which would cause a ValueError on int("").
                continue
            elif choice == "m":
                #The function "back_to_main_menu_function()"
                #will set the Boolean flags "is_in_submenu" and 
                #"is_in_sub_submenu" to "False", which will break the submenu
                #"while" loops and return to the main menu.
                back_to_main_menu_function(json_settings_dictionary)
                #A continue needs to be used, as we don't want 
                #the code below the "elif" statements to run,
                #which would cause a ValueError on int("m").
                continue
            elif choice == "b":
                is_in_sub_sub_submenu = False
                #A continue needs to be used, as we don't want 
                #the code below the "elif" statements to run,
                #which would cause a ValueError on int("b").
                continue
            elif choice == "r":
                #The "reset_to_default_setting()" function will reset the setting to its default value
                #found while accessing the value of the "json_default_settings_dictionary" dictionary 
                #with the key "setting_label_key". 
                #The "True" return value (if used) will break the submenu
                #loop and allow to return to the main menu or
                #the nested menu.
                json_settings_dictionary = reset_to_default_setting("Margins Filter Top Margin", json_settings_dictionary, 
                    json_default_settings_dictionary, json_settings_file_path_name)
                #A continue needs to be used, as we don't want 
                #the code below the "elif" statements to run,
                #which would cause a ValueError on int("r").
                continue
            elif choice == "t":
                #The "toggle_boolean_setting()" function will set the Boolean setting found while accessing
                #the "json_settings_dictionary" dictionary with the key "setting_label_key" to the opposite
                #value of the current setting ("True" if the current setting is "False" and vice-versa). 

                #The outcome string will be returned from the "toggle_boolean_setting()" 
                #function all if the default parameter Boolean flag "do_not_print_results" 
                #is set to "True", to ensure that the same state of the Filter is printed 
                #both at the top of the screen (current state) and in the confirmation string.
                json_settings_dictionary = toggle_boolean_setting("Margins Filter", json_settings_dictionary, 
                    json_settings_file_path_name)
                #A continue needs to be used, as we don't want 
                #the code below the "elif" statements to run,
                #which would cause a ValueError on int("r").
                continue
            elif choice == "q":
                quit_function()

            #The unit is removed (if provided)
            choice = re.sub(r"[ ]*%", "", choice)

            margin_height_percent = float(choice)
            if margin_height_percent >= 0:
                #The "set_numeric_setting()" function will set the value of the setting found while accessing
                #the "json_settings_dictionary" dictionary with the key "setting_label_key" to the provided
                #value ("setting_value"). 
                json_settings_dictionary = set_numeric_setting(margin_height_percent, "Margins Filter Top Margin", json_settings_dictionary, 
                    json_settings_file_path_name)
            else:
                input("\nInvalid choice, press any key to continue.")  
        except ValueError:
            input("\nInvalid choice, press any key to continue.")
    return json_settings_dictionary


#The "set_margins_filter_bottom_margin()" function will set the bottom margin for the margins filter.
#It will also allow the user to toggle the "Margins Filter" on or off.
def set_margins_filter_bottom_margin(json_settings_dictionary, json_default_settings_dictionary, json_settings_file_path_name):

    global is_in_sub_sub_submenu
    is_in_sub_sub_submenu = True

    while is_in_sub_sub_submenu:
        try:
            #The "clear_screen()" function will clear the CLI screen
            #using the appropriate command depending on the operating system.
            clear_screen()

            print("=== Set Margins Filter Bottom Margin ===\n\n")

            #The function "get_terminal_dimensions()" will return the number of columns 
            #and rows in the console, to allow to properly format the text and dividers.
            columns, lines = get_terminal_dimensions()

            textwrapped_instructions_string = textwrap.fill(bottom_margin_height_percent_comment_string, width=columns)
            textwrapped_input_string = textwrap.fill(f"Enter the bottom margin setting (0% or higher), or select one of the above options:", width=columns)

            if (json_settings_dictionary["Margins Filter"]):
                print("Margins filter is currently turned ON (Default value).\n")
            else:
                print("Margins filter is currently turned OFF.\n")

            print(f"Current Setting: {json_settings_dictionary["Margins Filter Bottom Margin"]}% | Default: {json_default_settings_dictionary["Margins Filter Bottom Margin"]}%.\n")

            print(textwrapped_instructions_string)
            print(f"\n[t] Toggle Filter On/Off\n[r] Reset to the Default Setting\n[b] Margins Filter Menu\n[m] Main Menu\n[q] Quit\n")

            choice = input(textwrapped_input_string + " ").strip().lower()

            if choice == "":
                #A continue needs to be used, as we don't want 
                #the code below the "elif" statements to run,
                #which would cause a ValueError on int("").
                continue
            elif choice == "m":
                #The function "back_to_main_menu_function()"
                #will set the Boolean flags "is_in_submenu" and 
                #"is_in_sub_submenu" to "False", which will break the submenu
                #"while" loops and return to the main menu.
                back_to_main_menu_function(json_settings_dictionary)
                #A continue needs to be used, as we don't want 
                #the code below the "elif" statements to run,
                #which would cause a ValueError on int("m").
                continue
            elif choice == "b":
                is_in_sub_sub_submenu = False
                #A continue needs to be used, as we don't want 
                #the code below the "elif" statements to run,
                #which would cause a ValueError on int("b").
                continue
            elif choice == "r":
                #The "reset_to_default_setting()" function will reset the setting to its default value
                #found while accessing the value of the "json_default_settings_dictionary" dictionary 
                #with the key "setting_label_key". 
                #The "True" return value (if used) will break the submenu
                #loop and allow to return to the main menu or
                #the nested menu.
                json_settings_dictionary = reset_to_default_setting("Margins Filter Bottom Margin", json_settings_dictionary, 
                    json_default_settings_dictionary, json_settings_file_path_name)
                #A continue needs to be used, as we don't want 
                #the code below the "elif" statements to run,
                #which would cause a ValueError on int("r").
                continue
            elif choice == "t":
                #The "toggle_boolean_setting()" function will set the Boolean setting found while accessing
                #the "json_settings_dictionary" dictionary with the key "setting_label_key" to the opposite
                #value of the current setting ("True" if the current setting is "False" and vice-versa). 

                #The outcome string will be returned from the "toggle_boolean_setting()" 
                #function all if the default parameter Boolean flag "do_not_print_results" 
                #is set to "True", to ensure that the same state of the Filter is printed 
                #both at the top of the screen (current state) and in the confirmation string.
                json_settings_dictionary = toggle_boolean_setting("Margins Filter", json_settings_dictionary, 
                    json_settings_file_path_name)
                #A continue needs to be used, as we don't want 
                #the code below the "elif" statements to run,
                #which would cause a ValueError on int("r").
                continue
            elif choice == "q":
                quit_function()

            #The unit is removed (if provided)
            choice = re.sub(r"[ ]*%", "", choice)

            margin_height_percent = float(choice)
            if margin_height_percent >= 0:
                #The "set_numeric_setting()" function will set the value of the setting found while accessing
                #the "json_settings_dictionary" dictionary with the key "setting_label_key" to the provided
                #value ("setting_value"). 
                json_settings_dictionary = set_numeric_setting(margin_height_percent, "Margins Filter Bottom Margin", json_settings_dictionary, 
                    json_settings_file_path_name)
            else:
                input("\nInvalid choice, press any key to continue.")   
        except ValueError:
            input("\nInvalid choice, press any key to continue.")
    return json_settings_dictionary


#The "margins_filter_menu()" function will run a "while is_in_submenu"
#loop that will allow the user to navigate the menu, and the loop will 
#be broken out of when they select the "Quit" option.
def margins_filter_menu(json_settings_dictionary, json_default_settings_dictionary, json_settings_file_path_name):

    global is_in_sub_submenu
    is_in_sub_submenu = True

    while is_in_sub_submenu:
        #The "clear_screen()" function will clear the CLI screen
        #using the appropriate command depending on the operating system.
        clear_screen()

        margins_filter_menu_actions_dict = {
        "1": ["Set Margins Filter Multiplier", set_margins_filter_multiplier, (json_settings_dictionary, json_default_settings_dictionary, json_settings_file_path_name)],
        "2": ["Set Left Margin", set_margins_filter_left_margin, (json_settings_dictionary, json_default_settings_dictionary, json_settings_file_path_name)],
        "3": ["Set Right Margin", set_margins_filter_right_margin, (json_settings_dictionary, json_default_settings_dictionary, json_settings_file_path_name)],
        "4": ["Set Top Margin", set_margins_filter_top_margin, (json_settings_dictionary, json_default_settings_dictionary, json_settings_file_path_name)],
        "5": ["Set Bottom Margin", set_margins_filter_bottom_margin, (json_settings_dictionary, json_default_settings_dictionary, json_settings_file_path_name)],
        "t": ["Toggle Filter On/Off", toggle_boolean_setting, ("Margins Filter", json_settings_dictionary, json_settings_file_path_name)],
        "b": ["Filters Menu", back_to_submenu_function, (json_settings_dictionary,)],
        "m": ["Main Menu", back_to_main_menu_function, (json_settings_dictionary,)],
        "q": ["Quit", quit_function, ()]}

        #The function "textwrap_action_strings_in_menu_action_dict()", which takes in 
        #a menu action dictionary comprised of one character keys and values made up
        #of a three-member tuple (action string, function, function arguments).
        #The action strings ("value[0]") will be textwrapped and the modified
        #dictionary will be returned.
        margins_filter_menu_actions_dict = textwrap_action_strings_in_menu_action_dict(margins_filter_menu_actions_dict)

        print("=== Margins Filter Menu ===\n\n")

        if (json_settings_dictionary["Margins Filter"]):
            print("Margins filter is currently turned ON (Default value).\n")
        else:
            print("Margins filter is currently turned OFF.\n")

        #The function "run_menu" will retrieve and call the function
        #at the appropriate choice key in the "menu_actions_dict"
        json_settings_dictionary = run_menu(margins_filter_menu_actions_dict, json_settings_dictionary)
    return json_settings_dictionary


#The "set_initial_page_color_filter_multiplier()" function will set the modifier for the page color filter.
def set_full_page_filter_multiplier(json_settings_dictionary, json_default_settings_dictionary, json_settings_file_path_name):

    global is_in_sub_submenu
    is_in_sub_submenu = True

    while is_in_sub_submenu:
        try:
            #The "clear_screen()" function will clear the CLI screen
            #using the appropriate command depending on the operating system.
            clear_screen()

            print("=== Set Full-Page Filter Multiplier ===\n\n")

            #The function "get_terminal_dimensions()" will return the number of columns 
            #and rows in the console, to allow to properly format the text and dividers.
            columns, lines = get_terminal_dimensions()

            textwrapped_instructions_string = textwrap.fill(number_of_standard_deviations_for_filtering_splotches_entire_page_comment_string, width=columns)
            textwrapped_input_string = textwrap.fill(f"Enter the value of the multiplier (-3.00 to +3.00), or select one of the above options:", width=columns)

            if (json_settings_dictionary["Full-Page Filter"]):
                print("Full-page filter is currently turned ON (Default value).\n")
            else:
                print("Full-page filter is currently turned OFF.\n")

            print(f"Current Setting: {json_settings_dictionary["Full-Page Filter Multiplier"]} | Default: {json_default_settings_dictionary["Full-Page Filter Multiplier"]}.\n")

            print(textwrapped_instructions_string)
            print(f"\n[t] Toggle Filter On/Off\n[r] Reset to the Default Setting\n[b] Filter Settings Menu\n[m] Main Menu\n[q] Quit\n")

            choice = input(textwrapped_input_string + " ").strip().lower()

            if choice == "":
                #A continue needs to be used, as we don't want 
                #the code below the "elif" statements to run,
                #which would cause a ValueError on int("").
                continue
            elif choice == "m":
                #The function "back_to_main_menu_function()"
                #will set the Boolean flags "is_in_submenu" and 
                #"is_in_sub_submenu" to "False", which will break the submenu
                #"while" loops and return to the main menu.
                back_to_main_menu_function(json_settings_dictionary)
                #A continue needs to be used, as we don't want 
                #the code below the "elif" statements to run,
                #which would cause a ValueError on int("m").
                continue
            elif choice == "b":
                is_in_sub_submenu = False
                #A continue needs to be used, as we don't want 
                #the code below the "elif" statements to run,
                #which would cause a ValueError on int("b").
                continue
            elif choice == "r":
                #The "reset_to_default_setting()" function will reset the setting to its default value
                #found while accessing the value of the "json_default_settings_dictionary" dictionary 
                #with the key "setting_label_key". 
                #The "True" return value (if used) will break the submenu
                #loop and allow to return to the main menu or
                #the nested menu.
                json_settings_dictionary = reset_to_default_setting("Full-Page Filter Multiplier", json_settings_dictionary, 
                    json_default_settings_dictionary, json_settings_file_path_name)
                #A continue needs to be used, as we don't want 
                #the code below the "elif" statements to run,
                #which would cause a ValueError on int("r").
                continue
            elif choice == "t":
                #The "toggle_boolean_setting()" function will set the Boolean setting found while accessing
                #the "json_settings_dictionary" dictionary with the key "setting_label_key" to the opposite
                #value of the current setting ("True" if the current setting is "False" and vice-versa). 

                #The outcome string will be returned from the "toggle_boolean_setting()" 
                #function all if the default parameter Boolean flag "do_not_print_results" 
                #is set to "True", to ensure that the same state of the Filter is printed 
                #both at the top of the screen (current state) and in the confirmation string.
                json_settings_dictionary = toggle_boolean_setting("Full-Page Filter", json_settings_dictionary, 
                    json_settings_file_path_name)
                #A continue needs to be used, as we don't want 
                #the code below the "elif" statements to run,
                #which would cause a ValueError on int("r").
                continue
            elif choice == "q":
                quit_function()

            full_page_filter_multiplier = float(choice)
            if -3.0 <= full_page_filter_multiplier <= 3.0:
                #The "set_numeric_setting()" function will set the value of the setting found while accessing
                #the "json_settings_dictionary" dictionary with the key "setting_label_key" to the provided
                #value ("setting_value"). 
                json_settings_dictionary = set_numeric_setting(full_page_filter_multiplier, "Full-Page Filter Multiplier", json_settings_dictionary, 
                    json_settings_file_path_name)
            else:
                input("\nInvalid choice, press any key to continue.")
        except ValueError:
            input("\nInvalid choice, press any key to continue.")
    return json_settings_dictionary


#The "filter_settings_menu()" function will run a "while is_in_submenu"
#loop that will allow the user to navigate the menu, and the loop will 
#be broken out of when they select the "Quit" option.
def filter_settings_menu(json_settings_dictionary, json_default_settings_dictionary, json_settings_file_path_name):

    global is_in_submenu
    is_in_submenu = True

    while is_in_submenu:

        filter_settings_menu_actions_dict = {
        "1": ["Initial Page Color Filter (Required. Removes the background page color)", page_color_filter_menu, (json_settings_dictionary, json_default_settings_dictionary, json_settings_file_path_name)],
        "2": ["Margins Filter (Recommended. Helps to properly crop the pages)", margins_filter_menu, (json_settings_dictionary, json_default_settings_dictionary, json_settings_file_path_name)],
        "3": ["Full-Page Filter (Optional. Use if any blotches remain in the center of the pages after the 'Initial Page Color Filter' step)", set_full_page_filter_multiplier, (json_settings_dictionary, json_default_settings_dictionary, json_settings_file_path_name)],
        "m": ["Main Menu", back_to_main_menu_function, (json_settings_dictionary,)],
        "q": ["Quit", quit_function, ()]}

        #The function "textwrap_action_strings_in_menu_action_dict()", which takes in 
        #a menu action dictionary comprised of one character keys and values made up
        #of a three-member tuple (action string, function, function arguments).
        #The action strings ("value[0]") will be textwrapped and the modified
        #dictionary will be returned.
        filter_settings_menu_actions_dict = textwrap_action_strings_in_menu_action_dict(filter_settings_menu_actions_dict)

        #The "clear_screen()" function will clear the CLI screen
        #using the appropriate command depending on the operating system.
        clear_screen()

        print("=== Filters Menu ===\n\n")

        #The function "run_menu" will retrieve and call the function
        #at the appropriate choice key in the "menu_actions_dict"
        json_settings_dictionary = run_menu(filter_settings_menu_actions_dict, json_settings_dictionary)
    return json_settings_dictionary


#The "set_left_right_kernel_size()" function will set the kernel size for the horizontal crop.
def set_left_right_kernel_size(json_settings_dictionary, json_default_settings_dictionary, json_settings_file_path_name):

    global is_in_sub_sub_submenu
    is_in_sub_sub_submenu = True

    while is_in_sub_sub_submenu:
        try:
            #The "clear_screen()" function will clear the CLI screen
            #using the appropriate command depending on the operating system.
            clear_screen()

            print("=== Set Left-Right Kernel Size ===\n\n")

            #The function "get_terminal_dimensions()" will return the number of columns 
            #and rows in the console, to allow to properly format the text and dividers.
            columns, lines = get_terminal_dimensions()

            textwrapped_instructions_string = textwrap.fill(horizontal_crop_kernel_size_height_percent_comment_string, width=columns)
            textwrapped_input_string = textwrap.fill(f"Enter the left-right kernel size (greater than 0%), or select one of the above options:", width=columns)

            if (json_settings_dictionary["Auto-Cropping"]):
                print("Auto-Cropping is currently turned ON (Default value).\n")
            else:
                print("Auto-Cropping filter is currently turned OFF.\n")

            if (json_settings_dictionary["Auto-Padding"]):
                print("Auto-Padding is currently turned ON (Default value).\n")
            else:
                print("Auto-Padding is currently turned OFF.\n")

            print(f"Current Setting: {json_settings_dictionary["Left-Right Kernel Size"]}% | Default: {json_default_settings_dictionary["Left-Right Kernel Size"]}%.\n")

            print(textwrapped_instructions_string)
            print(f"\n[t] Toggle Auto-Cropping On/Off\n[p] Toggle Auto-Padding On/Off\n[r] Reset to the Default Setting\n[b] Left-Right Crop Settings Menu\n[m] Main Menu\n[q] Quit\n")

            choice = input(textwrapped_input_string + " ").strip().lower()

            if choice == "":
                #A continue needs to be used, as we don't want 
                #the code below the "elif" statements to run,
                #which would cause a ValueError on int("").
                continue
            elif choice == "m":
                #The function "back_to_main_menu_function()"
                #will set the Boolean flags "is_in_submenu" and 
                #"is_in_sub_submenu" to "False", which will break the submenu
                #"while" loops and return to the main menu.
                back_to_main_menu_function(json_settings_dictionary)
                #A continue needs to be used, as we don't want 
                #the code below the "elif" statements to run,
                #which would cause a ValueError on int("m").
                continue
            elif choice == "b":
                is_in_sub_sub_submenu = False
                #A continue needs to be used, as we don't want 
                #the code below the "elif" statements to run,
                #which would cause a ValueError on int("b").
                continue
            elif choice == "r":
                #The "reset_to_default_setting()" function will reset the setting to its default value
                #found while accessing the value of the "json_default_settings_dictionary" dictionary 
                #with the key "setting_label_key". 
                #The "True" return value (if used) will break the submenu
                #loop and allow to return to the main menu or
                #the nested menu.
                json_settings_dictionary = reset_to_default_setting("Left-Right Kernel Size", json_settings_dictionary, 
                    json_default_settings_dictionary, json_settings_file_path_name)
                #A continue needs to be used, as we don't want 
                #the code below the "elif" statements to run,
                #which would cause a ValueError on int("r").
                continue
            elif choice == "t":
                #The "toggle_boolean_setting()" function will set the Boolean setting found while accessing
                #the "json_settings_dictionary" dictionary with the key "setting_label_key" to the opposite
                #value of the current setting ("True" if the current setting is "False" and vice-versa). 

                #The outcome string will be returned from the "toggle_boolean_setting()" 
                #function all if the default parameter Boolean flag "do_not_print_results" 
                #is set to "True", to ensure that the same state of the Filter is printed 
                #both at the top of the screen (current state) and in the confirmation string.
                json_settings_dictionary = toggle_boolean_setting("Auto-Cropping", json_settings_dictionary, 
                    json_settings_file_path_name)
                #A continue needs to be used, as we don't want 
                #the code below the "elif" statements to run,
                #which would cause a ValueError on int("r").
                continue
            elif choice == "p":
                #The "toggle_boolean_setting()" function will set the Boolean setting found while accessing
                #the "json_settings_dictionary" dictionary with the key "setting_label_key" to the opposite
                #value of the current setting ("True" if the current setting is "False" and vice-versa). 

                #The outcome string will be returned from the "toggle_boolean_setting()" 
                #function all if the default parameter Boolean flag "do_not_print_results" 
                #is set to "True", to ensure that the same state of the Filter is printed 
                #both at the top of the screen (current state) and in the confirmation string.
                json_settings_dictionary = toggle_boolean_setting("Auto-Padding", json_settings_dictionary, 
                    json_settings_file_path_name) 
                #A continue needs to be used, as we don't want 
                #the code below the "elif" statements to run,
                #which would cause a ValueError on int("r").
                continue
            elif choice == "q":
                quit_function()

            #The unit is removed (if provided)
            choice = re.sub(r"[ ]*%", "", choice)

            kernel_size_height_percent = float(choice)
            if kernel_size_height_percent > 0:
                #The "set_numeric_setting()" function will set the value of the setting found while accessing
                #the "json_settings_dictionary" dictionary with the key "setting_label_key" to the provided
                #value ("setting_value"). 
                json_settings_dictionary = set_numeric_setting(kernel_size_height_percent, "Left-Right Kernel Size", json_settings_dictionary, 
                    json_settings_file_path_name)
            else:
                input("\nInvalid choice, press any key to continue.")
        except ValueError:
            input("\nInvalid choice, press any key to continue.")
    return json_settings_dictionary


#The "set_left_right_kernel_radius()" function will set the kernel radius for the horizontal crop.
def set_left_right_kernel_radius(json_settings_dictionary, json_default_settings_dictionary, json_settings_file_path_name):

    global is_in_sub_sub_submenu
    is_in_sub_sub_submenu = True

    while is_in_sub_sub_submenu:
        try:
            #The "clear_screen()" function will clear the CLI screen
            #using the appropriate command depending on the operating system.
            clear_screen()

            print("=== Set Left-Right Kernel Radius ===\n\n")

            #The function "get_terminal_dimensions()" will return the number of columns 
            #and rows in the console, to allow to properly format the text and dividers.
            columns, lines = get_terminal_dimensions()

            textwrapped_instructions_string = textwrap.fill(horizontal_crop_kernel_radius_kernel_size_percent_comment_string, width=columns)
            textwrapped_input_string = textwrap.fill(f"Enter the left-right kernel radius (greater than 0%), or select one of the above options:", width=columns)

            if (json_settings_dictionary["Auto-Cropping"]):
                print("Auto-Cropping is currently turned ON (Default value).\n")
            else:
                print("Auto-Cropping filter is currently turned OFF.\n")

            if (json_settings_dictionary["Auto-Padding"]):
                print("Auto-Padding is currently turned ON (Default value).\n")
            else:
                print("Auto-Padding is currently turned OFF.\n")

            print(f"Current Setting: {json_settings_dictionary["Left-Right Kernel Radius"]}% | Default: {json_default_settings_dictionary["Left-Right Kernel Radius"]}%.\n")

            print(textwrapped_instructions_string)
            print(f"\n[t] Toggle Auto-Cropping On/Off\n[p] Toggle Auto-Padding On/Off\n[r] Reset to the Default Setting\n[b] Left-Right Crop Settings Menu\n[m] Main Menu\n[q] Quit\n")

            choice = input(textwrapped_input_string + " ").strip().lower()

            if choice == "":
                #A continue needs to be used, as we don't want 
                #the code below the "elif" statements to run,
                #which would cause a ValueError on int("").
                continue
            elif choice == "m":
                #The function "back_to_main_menu_function()"
                #will set the Boolean flags "is_in_submenu" and 
                #"is_in_sub_submenu" to "False", which will break the submenu
                #"while" loops and return to the main menu.
                back_to_main_menu_function(json_settings_dictionary)
                #A continue needs to be used, as we don't want 
                #the code below the "elif" statements to run,
                #which would cause a ValueError on int("m").
                continue
            elif choice == "b":
                is_in_sub_sub_submenu = False
                #A continue needs to be used, as we don't want 
                #the code below the "elif" statements to run,
                #which would cause a ValueError on int("b").
                continue
            elif choice == "r":
                #The "reset_to_default_setting()" function will reset the setting to its default value
                #found while accessing the value of the "json_default_settings_dictionary" dictionary 
                #with the key "setting_label_key". 
                #The "True" return value (if used) will break the submenu
                #loop and allow to return to the main menu or
                #the nested menu.
                json_settings_dictionary = reset_to_default_setting("Left-Right Kernel Radius", json_settings_dictionary, 
                    json_default_settings_dictionary, json_settings_file_path_name)
                #A continue needs to be used, as we don't want 
                #the code below the "elif" statements to run,
                #which would cause a ValueError on int("r").
                continue
            elif choice == "t":
                #The "toggle_boolean_setting()" function will set the Boolean setting found while accessing
                #the "json_settings_dictionary" dictionary with the key "setting_label_key" to the opposite
                #value of the current setting ("True" if the current setting is "False" and vice-versa). 

                #The outcome string will be returned from the "toggle_boolean_setting()" 
                #function all if the default parameter Boolean flag "do_not_print_results" 
                #is set to "True", to ensure that the same state of the Filter is printed 
                #both at the top of the screen (current state) and in the confirmation string.
                json_settings_dictionary = toggle_boolean_setting("Auto-Cropping", json_settings_dictionary, 
                    json_settings_file_path_name)
                #A continue needs to be used, as we don't want 
                #the code below the "elif" statements to run,
                #which would cause a ValueError on int("r").
                continue
            elif choice == "p":
                #The "toggle_boolean_setting()" function will set the Boolean setting found while accessing
                #the "json_settings_dictionary" dictionary with the key "setting_label_key" to the opposite
                #value of the current setting ("True" if the current setting is "False" and vice-versa). 

                #The outcome string will be returned from the "toggle_boolean_setting()" 
                #function all if the default parameter Boolean flag "do_not_print_results" 
                #is set to "True", to ensure that the same state of the Filter is printed 
                #both at the top of the screen (current state) and in the confirmation string.
                json_settings_dictionary = toggle_boolean_setting("Auto-Padding", json_settings_dictionary, 
                    json_settings_file_path_name) 
                #A continue needs to be used, as we don't want 
                #the code below the "elif" statements to run,
                #which would cause a ValueError on int("r").
                continue
            elif choice == "q":
                quit_function()

            #The unit is removed (if provided)
            choice = re.sub(r"[ ]*%", "", choice)

            kernel_radius_kernel_size_percent = float(choice)
            if kernel_radius_kernel_size_percent > 0:
                #The "set_numeric_setting()" function will set the value of the setting found while accessing
                #the "json_settings_dictionary" dictionary with the key "setting_label_key" to the provided
                #value ("setting_value"). 
                json_settings_dictionary = set_numeric_setting(kernel_radius_kernel_size_percent, "Left-Right Kernel Radius", json_settings_dictionary, 
                    json_settings_file_path_name)
            else:
                input("\nInvalid choice, press any key to continue.")  
        except ValueError:
            input("\nInvalid choice, press any key to continue.")
    return json_settings_dictionary


#The "set_left_right_safe_margin()" function will set the left-right safe margin size for the horizontal crop.
def set_left_right_safe_margin(json_settings_dictionary, json_default_settings_dictionary, json_settings_file_path_name):

    global is_in_sub_sub_submenu
    is_in_sub_sub_submenu = True

    while is_in_sub_sub_submenu:
        try:
            #The "clear_screen()" function will clear the CLI screen
            #using the appropriate command depending on the operating system.
            clear_screen()

            print("=== Set Left-Right Safe Margin ===\n\n")

            #The function "get_terminal_dimensions()" will return the number of columns 
            #and rows in the console, to allow to properly format the text and dividers.
            columns, lines = get_terminal_dimensions()

            textwrapped_instructions_string = textwrap.fill(horizontal_crop_margin_buffer_width_percentage_comment_string, width=columns)
            textwrapped_input_string = textwrap.fill(f"Enter the left-right safe margin (0% or higher), or select one of the above options:", width=columns)

            if (json_settings_dictionary["Auto-Cropping"]):
                print("Auto-Cropping is currently turned ON (Default value).\n")
            else:
                print("Auto-Cropping filter is currently turned OFF.\n")

            if (json_settings_dictionary["Auto-Padding"]):
                print("Auto-Padding is currently turned ON (Default value).\n")
            else:
                print("Auto-Padding is currently turned OFF.\n")

            print(f"Current Setting: {json_settings_dictionary["Left-Right Safe Margin Size"]}% | Default: {json_default_settings_dictionary["Left-Right Safe Margin Size"]}%.\n")

            print(textwrapped_instructions_string)
            print(f"\n[t] Toggle Auto-Cropping On/Off\n[p] Toggle Auto-Padding On/Off\n[r] Reset to the Default Setting\n[b] Left-Right Crop Settings Menu\n[m] Main Menu\n[q] Quit\n")

            choice = input(textwrapped_input_string + " ").strip().lower()

            if choice == "":
                #A continue needs to be used, as we don't want 
                #the code below the "elif" statements to run,
                #which would cause a ValueError on int("").
                continue
            elif choice == "m":
                #The function "back_to_main_menu_function()"
                #will set the Boolean flags "is_in_submenu" and 
                #"is_in_sub_submenu" to "False", which will break the submenu
                #"while" loops and return to the main menu.
                back_to_main_menu_function(json_settings_dictionary)
                #A continue needs to be used, as we don't want 
                #the code below the "elif" statements to run,
                #which would cause a ValueError on int("m").
                continue
            elif choice == "b":
                is_in_sub_sub_submenu = False
                #A continue needs to be used, as we don't want 
                #the code below the "elif" statements to run,
                #which would cause a ValueError on int("b").
                continue
            elif choice == "r":
                #The "reset_to_default_setting()" function will reset the setting to its default value
                #found while accessing the value of the "json_default_settings_dictionary" dictionary 
                #with the key "setting_label_key". 
                #The "True" return value (if used) will break the submenu
                #loop and allow to return to the main menu or
                #the nested menu.
                json_settings_dictionary = reset_to_default_setting("Left-Right Safe Margin Size", json_settings_dictionary, 
                    json_default_settings_dictionary, json_settings_file_path_name)
                #A continue needs to be used, as we don't want 
                #the code below the "elif" statements to run,
                #which would cause a ValueError on int("r").
                continue
            elif choice == "t":
                #The "toggle_boolean_setting()" function will set the Boolean setting found while accessing
                #the "json_settings_dictionary" dictionary with the key "setting_label_key" to the opposite
                #value of the current setting ("True" if the current setting is "False" and vice-versa). 

                #The outcome string will be returned from the "toggle_boolean_setting()" 
                #function all if the default parameter Boolean flag "do_not_print_results" 
                #is set to "True", to ensure that the same state of the Filter is printed 
                #both at the top of the screen (current state) and in the confirmation string.
                json_settings_dictionary = toggle_boolean_setting("Auto-Cropping", json_settings_dictionary, 
                    json_settings_file_path_name)
                #A continue needs to be used, as we don't want 
                #the code below the "elif" statements to run,
                #which would cause a ValueError on int("r").
                continue
            elif choice == "p":
                #The "toggle_boolean_setting()" function will set the Boolean setting found while accessing
                #the "json_settings_dictionary" dictionary with the key "setting_label_key" to the opposite
                #value of the current setting ("True" if the current setting is "False" and vice-versa). 

                #The outcome string will be returned from the "toggle_boolean_setting()" 
                #function all if the default parameter Boolean flag "do_not_print_results" 
                #is set to "True", to ensure that the same state of the Filter is printed 
                #both at the top of the screen (current state) and in the confirmation string.
                json_settings_dictionary = toggle_boolean_setting("Auto-Padding", json_settings_dictionary, 
                    json_settings_file_path_name) 
                #A continue needs to be used, as we don't want 
                #the code below the "elif" statements to run,
                #which would cause a ValueError on int("r").
                continue
            elif choice == "q":
                quit_function()

            #The unit is removed (if provided)
            choice = re.sub(r"[ ]*%", "", choice)

            safe_margin_size_width_percent = float(choice)
            if safe_margin_size_width_percent >= 0:
                #The "set_numeric_setting()" function will set the value of the setting found while accessing
                #the "json_settings_dictionary" dictionary with the key "setting_label_key" to the provided
                #value ("setting_value"). 
                json_settings_dictionary = set_numeric_setting(safe_margin_size_width_percent, "Left-Right Safe Margin Size", json_settings_dictionary, 
                    json_settings_file_path_name)
            else:
                input("\nInvalid choice, press any key to continue.") 
        except ValueError:
            input("\nInvalid choice, press any key to continue.")
    return json_settings_dictionary


#The "left_right_crop_settings_menu()" function will run a "while is_in_submenu"
#loop that will allow the user to navigate the menu, and the loop will 
#be broken out of when they select the "Quit" option.
def left_right_crop_settings_menu(json_settings_dictionary, json_default_settings_dictionary, json_settings_file_path_name):

    global is_in_sub_submenu
    is_in_sub_submenu = True

    while is_in_sub_submenu:
        #The "clear_screen()" function will clear the CLI screen
        #using the appropriate command depending on the operating system.
        clear_screen()

        left_right_auto_crop_settings_menu_actions_dict = {
        "1": ["Set Left-Right Kernel Size (Total span of the horizontal text-edge search area)", set_left_right_kernel_size, (json_settings_dictionary, json_default_settings_dictionary, json_settings_file_path_name)],
        "2": ["Set Left-Right Kernel Radius (Max gap distance for merging separate text fragments)", set_left_right_kernel_radius, (json_settings_dictionary, json_default_settings_dictionary, json_settings_file_path_name)],
        "3": ["Set Left-Right Safe Margin Size (Used for expanding the crop to maintain a safe margin around the text)", set_left_right_safe_margin, (json_settings_dictionary, json_default_settings_dictionary, json_settings_file_path_name)],
        "t": ["Toggle Auto-Cropping On/Off (Automatically crops the horizontal and vertical margins)", toggle_boolean_setting, ("Auto-Cropping", json_settings_dictionary, json_settings_file_path_name)],
        "p": ["Toggle Auto-Padding On/Off (Pads all of the cropped pages so that they end up with the same dimensions)", toggle_boolean_setting, ("Auto-Padding", json_settings_dictionary, json_settings_file_path_name)],
        "b": ["Auto-Cropping Menu", back_to_submenu_function, (json_settings_dictionary,)],
        "m": ["Main Menu", back_to_main_menu_function, (json_settings_dictionary,)],
        "q": ["Quit", quit_function, ()]}

        #The function "textwrap_action_strings_in_menu_action_dict()", which takes in 
        #a menu action dictionary comprised of one character keys and values made up
        #of a three-member tuple (action string, function, function arguments).
        #The action strings ("value[0]") will be textwrapped and the modified
        #dictionary will be returned.
        left_right_auto_crop_settings_menu_actions_dict = textwrap_action_strings_in_menu_action_dict(left_right_auto_crop_settings_menu_actions_dict)

        print("=== Left-Right Crop Settings Menu ===\n\n")

        if (json_settings_dictionary["Auto-Cropping"]):
            print("Auto-Cropping is currently turned ON (Default value).\n")
        else:
            print("Auto-Cropping filter is currently turned OFF.\n")

        if (json_settings_dictionary["Auto-Padding"]):
            print("Auto-Padding is currently turned ON (Default value).\n")
        else:
            print("Auto-Padding is currently turned OFF.\n")

        #The function "get_terminal_dimensions()" will return the number of columns 
        #and rows in the console, to allow to properly format the text and dividers.
        columns, lines = get_terminal_dimensions()

        print(textwrap.fill(auto_cropping_comment_string.replace(" (default setting: True)", ""), width=columns) + "\n")

        print(textwrap.fill(auto_padding_mode_comment_string.replace(" (default setting: True)", ""), width=columns) + "\n")

        #The function "run_menu" will retrieve and call the function
        #at the appropriate choice key in the "menu_actions_dict"
        json_settings_dictionary = run_menu(left_right_auto_crop_settings_menu_actions_dict, json_settings_dictionary)
    return json_settings_dictionary


#The "set_top_bottom_kernel_size()" function will set the kernel size for the vertical crop.
def set_top_bottom_kernel_size(json_settings_dictionary, json_default_settings_dictionary, json_settings_file_path_name):

    global is_in_sub_sub_submenu
    is_in_sub_sub_submenu = True

    while is_in_sub_sub_submenu:
        try:
            #The "clear_screen()" function will clear the CLI screen
            #using the appropriate command depending on the operating system.
            clear_screen()

            print("=== Set Top-Bottom Kernel Size ===\n\n")

            #The function "get_terminal_dimensions()" will return the number of columns 
            #and rows in the console, to allow to properly format the text and dividers.
            columns, lines = get_terminal_dimensions()

            textwrapped_instructions_string = textwrap.fill(vertical_crop_kernel_size_height_percent_comment_string, width=columns)
            textwrapped_input_string = textwrap.fill(f"Enter the top-bottom kernel size (greater than 0%), or select one of the above options:", width=columns)

            if (json_settings_dictionary["Auto-Cropping"]):
                print("Auto-Cropping is currently turned ON (Default value).\n")
            else:
                print("Auto-Cropping filter is currently turned OFF.\n")

            if (json_settings_dictionary["Auto-Padding"]):
                print("Auto-Padding is currently turned ON (Default value).\n")
            else:
                print("Auto-Padding is currently turned OFF.\n")

            print(f"Current Setting: {json_settings_dictionary["Top-Bottom Kernel Size"]}% | Default: {json_default_settings_dictionary["Top-Bottom Kernel Size"]}%.\n")

            print(textwrapped_instructions_string)
            print(f"\n[t] Toggle Auto-Cropping On/Off\n[p] Toggle Auto-Padding On/Off\n[r] Reset to the Default Setting\n[b] Top-Bottom Crop Settings Menu\n[m] Main Menu\n[q] Quit\n")

            choice = input(textwrapped_input_string + " ").strip().lower()

            if choice == "":
                #A continue needs to be used, as we don't want 
                #the code below the "elif" statements to run,
                #which would cause a ValueError on int("").
                continue
            elif choice == "m":
                #The function "back_to_main_menu_function()"
                #will set the Boolean flags "is_in_submenu" and 
                #"is_in_sub_submenu" to "False", which will break the submenu
                #"while" loops and return to the main menu.
                back_to_main_menu_function(json_settings_dictionary)
                #A continue needs to be used, as we don't want 
                #the code below the "elif" statements to run,
                #which would cause a ValueError on int("m").
                continue
            elif choice == "b":
                is_in_sub_sub_submenu = False
                #A continue needs to be used, as we don't want 
                #the code below the "elif" statements to run,
                #which would cause a ValueError on int("b").
                continue
            elif choice == "r":
                #The "reset_to_default_setting()" function will reset the setting to its default value
                #found while accessing the value of the "json_default_settings_dictionary" dictionary 
                #with the key "setting_label_key". 
                #The "True" return value (if used) will break the submenu
                #loop and allow to return to the main menu or
                #the nested menu.
                json_settings_dictionary = reset_to_default_setting("Top-Bottom Kernel Size", json_settings_dictionary, 
                    json_default_settings_dictionary, json_settings_file_path_name)
                #A continue needs to be used, as we don't want 
                #the code below the "elif" statements to run,
                #which would cause a ValueError on int("r").
                continue
            elif choice == "t":
                #The "toggle_boolean_setting()" function will set the Boolean setting found while accessing
                #the "json_settings_dictionary" dictionary with the key "setting_label_key" to the opposite
                #value of the current setting ("True" if the current setting is "False" and vice-versa). 

                #The outcome string will be returned from the "toggle_boolean_setting()" 
                #function all if the default parameter Boolean flag "do_not_print_results" 
                #is set to "True", to ensure that the same state of the Filter is printed 
                #both at the top of the screen (current state) and in the confirmation string.
                json_settings_dictionary = toggle_boolean_setting("Auto-Cropping", json_settings_dictionary, 
                    json_settings_file_path_name)
                #A continue needs to be used, as we don't want 
                #the code below the "elif" statements to run,
                #which would cause a ValueError on int("r").
                continue
            elif choice == "p":
                #The "toggle_boolean_setting()" function will set the Boolean setting found while accessing
                #the "json_settings_dictionary" dictionary with the key "setting_label_key" to the opposite
                #value of the current setting ("True" if the current setting is "False" and vice-versa). 

                #The outcome string will be returned from the "toggle_boolean_setting()" 
                #function all if the default parameter Boolean flag "do_not_print_results" 
                #is set to "True", to ensure that the same state of the Filter is printed 
                #both at the top of the screen (current state) and in the confirmation string.
                json_settings_dictionary = toggle_boolean_setting("Auto-Padding", json_settings_dictionary, 
                    json_settings_file_path_name) 
                #A continue needs to be used, as we don't want 
                #the code below the "elif" statements to run,
                #which would cause a ValueError on int("r").
                continue
            elif choice == "q":
                quit_function()

            #The unit is removed (if provided)
            choice = re.sub(r"[ ]*%", "", choice)

            kernel_size_height_percent = float(choice)
            if kernel_size_height_percent > 0:
                #The "set_numeric_setting()" function will set the value of the setting found while accessing
                #the "json_settings_dictionary" dictionary with the key "setting_label_key" to the provided
                #value ("setting_value"). 
                json_settings_dictionary = set_numeric_setting(kernel_size_height_percent, "Top-Bottom Kernel Size", json_settings_dictionary, 
                    json_settings_file_path_name)
            else:
                input("\nInvalid choice, press any key to continue.") 
        except ValueError:
            input("\nInvalid choice, press any key to continue.")
    return json_settings_dictionary


#The "set_top_bottom_kernel_radius()" function will set the kernel radius for the vertical crop.
def set_top_bottom_kernel_radius(json_settings_dictionary, json_default_settings_dictionary, json_settings_file_path_name):

    global is_in_sub_sub_submenu
    is_in_sub_sub_submenu = True

    while is_in_sub_sub_submenu:
        try:
            #The "clear_screen()" function will clear the CLI screen
            #using the appropriate command depending on the operating system.
            clear_screen()

            print("=== Set Top-Bottom Kernel Radius ===\n\n")

            #The function "get_terminal_dimensions()" will return the number of columns 
            #and rows in the console, to allow to properly format the text and dividers.
            columns, lines = get_terminal_dimensions()

            textwrapped_instructions_string = textwrap.fill(vertical_crop_kernel_radius_kernel_size_percent_comment_string, width=columns)
            textwrapped_input_string = textwrap.fill(f"Enter the top-bottom kernel radius (greater than 0%), or select one of the above options:", width=columns)

            if (json_settings_dictionary["Auto-Cropping"]):
                print("Auto-Cropping is currently turned ON (Default value).\n")
            else:
                print("Auto-Cropping filter is currently turned OFF.\n")

            if (json_settings_dictionary["Auto-Padding"]):
                print("Auto-Padding is currently turned ON (Default value).\n")
            else:
                print("Auto-Padding is currently turned OFF.\n")

            print(f"Current Setting: {json_settings_dictionary["Top-Bottom Kernel Radius"]}% | Default: {json_default_settings_dictionary["Top-Bottom Kernel Radius"]}%.\n")

            print(textwrapped_instructions_string)
            print(f"\n[t] Toggle Auto-Cropping On/Off\n[p] Toggle Auto-Padding On/Off\n[r] Reset to the Default Setting\n[b] Top-Bottom Crop Settings Menu\n[m] Main Menu\n[q] Quit\n")

            choice = input(textwrapped_input_string + " ").strip().lower()

            if choice == "":
                #A continue needs to be used, as we don't want 
                #the code below the "elif" statements to run,
                #which would cause a ValueError on int("").
                continue
            elif choice == "m":
                #The function "back_to_main_menu_function()"
                #will set the Boolean flags "is_in_submenu" and 
                #"is_in_sub_submenu" to "False", which will break the submenu
                #"while" loops and return to the main menu.
                back_to_main_menu_function(json_settings_dictionary)
                #A continue needs to be used, as we don't want 
                #the code below the "elif" statements to run,
                #which would cause a ValueError on int("m").
                continue
            elif choice == "b":
                is_in_sub_sub_submenu = False
                #A continue needs to be used, as we don't want 
                #the code below the "elif" statements to run,
                #which would cause a ValueError on int("b").
                continue
            elif choice == "r":
                #The "reset_to_default_setting()" function will reset the setting to its default value
                #found while accessing the value of the "json_default_settings_dictionary" dictionary 
                #with the key "setting_label_key". 
                #The "True" return value (if used) will break the submenu
                #loop and allow to return to the main menu or
                #the nested menu.
                json_settings_dictionary = reset_to_default_setting("Top-Bottom Kernel Radius", json_settings_dictionary, 
                    json_default_settings_dictionary, json_settings_file_path_name)
                #A continue needs to be used, as we don't want 
                #the code below the "elif" statements to run,
                #which would cause a ValueError on int("r").
                continue
            elif choice == "t":
                #The "toggle_boolean_setting()" function will set the Boolean setting found while accessing
                #the "json_settings_dictionary" dictionary with the key "setting_label_key" to the opposite
                #value of the current setting ("True" if the current setting is "False" and vice-versa). 

                #The outcome string will be returned from the "toggle_boolean_setting()" 
                #function all if the default parameter Boolean flag "do_not_print_results" 
                #is set to "True", to ensure that the same state of the Filter is printed 
                #both at the top of the screen (current state) and in the confirmation string.
                json_settings_dictionary = toggle_boolean_setting("Auto-Cropping", json_settings_dictionary, 
                    json_settings_file_path_name)
                #A continue needs to be used, as we don't want 
                #the code below the "elif" statements to run,
                #which would cause a ValueError on int("r").
                continue
            elif choice == "p":
                #The "toggle_boolean_setting()" function will set the Boolean setting found while accessing
                #the "json_settings_dictionary" dictionary with the key "setting_label_key" to the opposite
                #value of the current setting ("True" if the current setting is "False" and vice-versa). 

                #The outcome string will be returned from the "toggle_boolean_setting()" 
                #function all if the default parameter Boolean flag "do_not_print_results" 
                #is set to "True", to ensure that the same state of the Filter is printed 
                #both at the top of the screen (current state) and in the confirmation string.
                json_settings_dictionary = toggle_boolean_setting("Auto-Padding", json_settings_dictionary, 
                    json_settings_file_path_name) 
                #A continue needs to be used, as we don't want 
                #the code below the "elif" statements to run,
                #which would cause a ValueError on int("r").
                continue
            elif choice == "q":
                quit_function()

            #The unit is removed (if provided)
            choice = re.sub(r"[ ]*%", "", choice)

            kernel_radius_kernel_size_percent = float(choice)
            if kernel_radius_kernel_size_percent > 0:
                #The "set_numeric_setting()" function will set the value of the setting found while accessing
                #the "json_settings_dictionary" dictionary with the key "setting_label_key" to the provided
                #value ("setting_value"). 
                json_settings_dictionary = set_numeric_setting(kernel_radius_kernel_size_percent, "Top-Bottom Kernel Radius", json_settings_dictionary, 
                    json_settings_file_path_name)
            else:
                input("\nInvalid choice, press any key to continue.")   
        except ValueError:
            input("\nInvalid choice, press any key to continue.")
    return json_settings_dictionary


#The "set_top_bottom_safe_margin()" function will set the top-bottom safe margin size for the vertical crop.
def set_top_bottom_safe_margin(json_settings_dictionary, json_default_settings_dictionary, json_settings_file_path_name):

    global is_in_sub_sub_submenu
    is_in_sub_sub_submenu = True

    while is_in_sub_sub_submenu:
        try:
            #The "clear_screen()" function will clear the CLI screen
            #using the appropriate command depending on the operating system.
            clear_screen()

            print("=== Set Top-Bottom Safe Margin ===\n\n")

            #The function "get_terminal_dimensions()" will return the number of columns 
            #and rows in the console, to allow to properly format the text and dividers.
            columns, lines = get_terminal_dimensions()

            textwrapped_instructions_string = textwrap.fill(vertical_crop_margin_buffer_height_percentage_comment_string, width=columns)
            textwrapped_input_string = textwrap.fill(f"Enter the top-bottom safe margin setting (0% or higher), or select one of the above options:", width=columns)

            if (json_settings_dictionary["Auto-Cropping"]):
                print("Auto-Cropping is currently turned ON (Default value).\n")
            else:
                print("Auto-Cropping filter is currently turned OFF.\n")

            if (json_settings_dictionary["Auto-Padding"]):
                print("Auto-Padding is currently turned ON (Default value).\n")
            else:
                print("Auto-Padding is currently turned OFF.\n")

            print(f"Current Setting: {json_settings_dictionary["Top-Bottom Safe Margin Size"]}% | Default: {json_default_settings_dictionary["Top-Bottom Safe Margin Size"]}%.\n")

            print(textwrapped_instructions_string)
            print(f"\n[t] Toggle Auto-Cropping On/Off\n[p] Toggle Auto-Padding On/Off\n[r] Reset to the Default Setting\n[b] Top-Bottom Crop Settings Menu\n[m] Main Menu\n[q] Quit\n")

            choice = input(textwrapped_input_string + " ").strip().lower()

            if choice == "":
                #A continue needs to be used, as we don't want 
                #the code below the "elif" statements to run,
                #which would cause a ValueError on int("").
                continue
            elif choice == "m":
                #The function "back_to_main_menu_function()"
                #will set the Boolean flags "is_in_submenu" and 
                #"is_in_sub_submenu" to "False", which will break the submenu
                #"while" loops and return to the main menu.
                back_to_main_menu_function(json_settings_dictionary)
                #A continue needs to be used, as we don't want 
                #the code below the "elif" statements to run,
                #which would cause a ValueError on int("m").
                continue
            elif choice == "b":
                is_in_sub_sub_submenu = False
                #A continue needs to be used, as we don't want 
                #the code below the "elif" statements to run,
                #which would cause a ValueError on int("b").
                continue
            elif choice == "r":
                #The "reset_to_default_setting()" function will reset the setting to its default value
                #found while accessing the value of the "json_default_settings_dictionary" dictionary 
                #with the key "setting_label_key". 
                #The "True" return value (if used) will break the submenu
                #loop and allow to return to the main menu or
                #the nested menu.
                json_settings_dictionary = reset_to_default_setting("Top-Bottom Safe Margin Size", json_settings_dictionary, 
                    json_default_settings_dictionary, json_settings_file_path_name)
                #A continue needs to be used, as we don't want 
                #the code below the "elif" statements to run,
                #which would cause a ValueError on int("r").
                continue
            elif choice == "t":
                #The "toggle_boolean_setting()" function will set the Boolean setting found while accessing
                #the "json_settings_dictionary" dictionary with the key "setting_label_key" to the opposite
                #value of the current setting ("True" if the current setting is "False" and vice-versa). 

                #The outcome string will be returned from the "toggle_boolean_setting()" 
                #function all if the default parameter Boolean flag "do_not_print_results" 
                #is set to "True", to ensure that the same state of the Filter is printed 
                #both at the top of the screen (current state) and in the confirmation string.
                json_settings_dictionary = toggle_boolean_setting("Auto-Cropping", json_settings_dictionary, 
                    json_settings_file_path_name)
                #A continue needs to be used, as we don't want 
                #the code below the "elif" statements to run,
                #which would cause a ValueError on int("r").
                continue
            elif choice == "p":
                #The "toggle_boolean_setting()" function will set the Boolean setting found while accessing
                #the "json_settings_dictionary" dictionary with the key "setting_label_key" to the opposite
                #value of the current setting ("True" if the current setting is "False" and vice-versa). 

                #The outcome string will be returned from the "toggle_boolean_setting()" 
                #function all if the default parameter Boolean flag "do_not_print_results" 
                #is set to "True", to ensure that the same state of the Filter is printed 
                #both at the top of the screen (current state) and in the confirmation string.
                json_settings_dictionary = toggle_boolean_setting("Auto-Padding", json_settings_dictionary, 
                    json_settings_file_path_name) 
                #A continue needs to be used, as we don't want 
                #the code below the "elif" statements to run,
                #which would cause a ValueError on int("r").
                continue   
            elif choice == "q":
                quit_function()

            #The unit is removed (if provided)
            choice = re.sub(r"[ ]*%", "", choice)

            safe_margin_size_width_percent = float(choice)
            if safe_margin_size_width_percent >= 0:
                #The "set_numeric_setting()" function will set the value of the setting found while accessing
                #the "json_settings_dictionary" dictionary with the key "setting_label_key" to the provided
                #value ("setting_value"). 
                json_settings_dictionary = set_numeric_setting(safe_margin_size_width_percent, "Top-Bottom Safe Margin Size", json_settings_dictionary, 
                    json_settings_file_path_name)
            else:
                input("\nInvalid choice, press any key to continue.") 
        except ValueError:
            input("\nInvalid choice, press any key to continue.")
    return json_settings_dictionary


#The "top_bottom_crop_settings_menu()" function will run a "while is_in_submenu"
#loop that will allow the user to navigate the menu, and the loop will 
#be broken out of when they select the "Quit" option.
def top_bottom_crop_settings_menu(json_settings_dictionary, json_default_settings_dictionary, json_settings_file_path_name):

    global is_in_sub_submenu
    is_in_sub_submenu = True

    while is_in_sub_submenu:
        #The "clear_screen()" function will clear the CLI screen
        #using the appropriate command depending on the operating system.
        clear_screen()

        top_bottom_auto_crop_settings_menu_actions_dict = {
        "1": ["Set Top-Bottom Kernel Size (Total span of the vertical text-edge search area)", set_top_bottom_kernel_size, (json_settings_dictionary, json_default_settings_dictionary, json_settings_file_path_name)],
        "2": ["Set Top-Bottom Kernel Radius (Max gap distance for merging separate text fragments)", set_top_bottom_kernel_radius, (json_settings_dictionary, json_default_settings_dictionary, json_settings_file_path_name)],
        "3": ["Set Top-Bottom Safe Margin Size (Used for expanding the crop to maintain a safe margin around the text)", set_top_bottom_safe_margin, (json_settings_dictionary, json_default_settings_dictionary, json_settings_file_path_name)],
        "t": ["Toggle Auto-Cropping On/Off (Automatically crops the horizontal and vertical margins)", toggle_boolean_setting, ("Auto-Cropping", json_settings_dictionary, json_settings_file_path_name)],
        "p": ["Toggle Auto-Padding On/Off (Pads all of the cropped pages so that they end up with the same dimensions)", toggle_boolean_setting, ("Auto-Padding", json_settings_dictionary, json_settings_file_path_name)],
        "b": ["Auto-Cropping Menu", back_to_submenu_function, (json_settings_dictionary,)],
        "m": ["Main Menu", back_to_main_menu_function, (json_settings_dictionary,)],
        "q": ["Quit", quit_function, ()]}

        #The function "textwrap_action_strings_in_menu_action_dict()", which takes in 
        #a menu action dictionary comprised of one character keys and values made up
        #of a three-member tuple (action string, function, function arguments).
        #The action strings ("value[0]") will be textwrapped and the modified
        #dictionary will be returned.
        top_bottom_auto_crop_settings_menu_actions_dict = textwrap_action_strings_in_menu_action_dict(top_bottom_auto_crop_settings_menu_actions_dict)

        print("=== Top-Bottom Crop Settings Menu ===\n\n")

        if (json_settings_dictionary["Auto-Cropping"]):
            print("Auto-Cropping is currently turned ON (Default value).\n")
        else:
            print("Auto-Cropping filter is currently turned OFF.\n")

        if (json_settings_dictionary["Auto-Padding"]):
            print("Auto-Padding is currently turned ON (Default value).\n")
        else:
            print("Auto-Padding is currently turned OFF.\n")

        #The function "get_terminal_dimensions()" will return the number of columns 
        #and rows in the console, to allow to properly format the text and dividers.
        columns, lines = get_terminal_dimensions()

        print(textwrap.fill(auto_cropping_comment_string.replace(" (default setting: True)", ""), width=columns) + "\n")

        print(textwrap.fill(auto_padding_mode_comment_string.replace(" (default setting: True)", ""), width=columns) + "\n")

        #The function "run_menu" will retrieve and call the function
        #at the appropriate choice key in the "menu_actions_dict"
        json_settings_dictionary = run_menu(top_bottom_auto_crop_settings_menu_actions_dict, json_settings_dictionary)
    return json_settings_dictionary


#The "auto_crop_settings_menu()" function will run a "while is_in_submenu"
#loop that will allow the user to navigate the menu, and the loop will 
#be broken out of when they select the "Quit" option.
def auto_crop_settings_menu(json_settings_dictionary, json_default_settings_dictionary, json_settings_file_path_name):

    global is_in_submenu
    is_in_submenu = True

    while is_in_submenu:
        #The "clear_screen()" function will clear the CLI screen
        #using the appropriate command depending on the operating system.
        clear_screen()

        auto_crop_settings_menu_actions_dict = {
        "1": ["Left-Right Cropping", left_right_crop_settings_menu, (json_settings_dictionary, json_default_settings_dictionary, json_settings_file_path_name)],
        "2": ["Top-Bottom Cropping", top_bottom_crop_settings_menu, (json_settings_dictionary, json_default_settings_dictionary, json_settings_file_path_name)],
        "t": ["Toggle Auto-Cropping On/Off (Automatically crops the horizontal and vertical margins)", toggle_boolean_setting, ("Auto-Cropping", json_settings_dictionary, json_settings_file_path_name)],
        "p": ["Toggle Auto-Padding On/Off (Pads all of the cropped pages so that they end up with the same dimensions)", toggle_boolean_setting, ("Auto-Padding", json_settings_dictionary, json_settings_file_path_name)],
        "m": ["Main Menu", back_to_main_menu_function, (json_settings_dictionary,)],
        "q": ["Quit", quit_function, ()]}

        #The function "textwrap_action_strings_in_menu_action_dict()", which takes in 
        #a menu action dictionary comprised of one character keys and values made up
        #of a three-member tuple (action string, function, function arguments).
        #The action strings ("value[0]") will be textwrapped and the modified
        #dictionary will be returned.
        auto_crop_settings_menu_actions_dict = textwrap_action_strings_in_menu_action_dict(auto_crop_settings_menu_actions_dict)

        print("=== Auto-Cropping Settings Menu ===\n\n")

        if (json_settings_dictionary["Auto-Cropping"]):
            print("Auto-Cropping is currently turned ON (Default value).\n")
        else:
            print("Auto-Cropping filter is currently turned OFF.\n")

        if (json_settings_dictionary["Auto-Padding"]):
            print("Auto-Padding is currently turned ON (Default value).\n")
        else:
            print("Auto-Padding is currently turned OFF.\n")

        #The function "get_terminal_dimensions()" will return the number of columns 
        #and rows in the console, to allow to properly format the text and dividers.
        columns, lines = get_terminal_dimensions()

        print(textwrap.fill(auto_cropping_comment_string.replace(" (default setting: True)", ""), width=columns) + "\n")

        print(textwrap.fill(auto_padding_mode_comment_string.replace(" (default setting: True)", ""), width=columns) + "\n")


        #The function "run_menu" will retrieve and call the function
        #at the appropriate choice key in the "menu_actions_dict"
        json_settings_dictionary = run_menu(auto_crop_settings_menu_actions_dict, json_settings_dictionary)
    return json_settings_dictionary


#The "reset_all_settings()" function will set the value "json_settings_dictionary"
#to that of "json_default_Settings_dictionary" and save the changes to the JSON file.
def reset_all_settings(json_settings_dictionary, json_default_settings_dictionary, json_settings_file_path_name):

    global is_in_submenu
    is_in_submenu = True

    while is_in_submenu:

        #The "clear_screen()" function will clear the CLI screen
        #using the appropriate command depending on the operating system.
        clear_screen()

        print("=== Reset All Settings ===\n\n")

        #The function "get_terminal_dimensions()" will return the number of columns 
        #and rows in the console, to allow to properly format the text and dividers.
        columns, lines = get_terminal_dimensions()
        textwrapped_input_string = textwrap.fill("Are you sure you want to reset all of the settings? Enter (y/n), or select one of the above options: ", width=columns)

        print(f"[m] Main Menu\n[q] Quit\n")

        choice = input(textwrapped_input_string + " ").strip().lower()
        if choice in ["", "n"]:
            #A continue needs to be used, as we don't want 
            #the code below the "elif" statements to run,
            #which would cause a ValueError on int("").
            continue
        elif choice == "m":
            #The function "back_to_main_menu_function()"
            #will set the Boolean flags "is_in_submenu" and 
            #"is_in_sub_submenu" to "False", which will break the submenu
            #"while" loops and return to the main menu.
            return json_settings_dictionary
        elif choice == "q":
            quit_function()
        elif choice == "y":           
            #A deep copy (since it contains a list of deleted pages) of 
            #"json_default_settings_dictionary" is made so as to avoid having
            #both "json_settings_dictionary" and "json_default_settings_dictionary"
            #pointing to the same address.
            json_settings_dictionary = copy.deepcopy(json_default_settings_dictionary)
            #The function "atomic_save()" will create a temporary JSON file with the updated changes.
            #If the files is created successfully, then the files will be swapped. If a problem is 
            #encountered, the temp file will be unlinked and an error log will be reported.
            atomic_save(json_settings_dictionary, json_settings_file_path_name)
            print("\nAll settings have successfully been reset to their default values.")
            input("\nPress any key continue.")
        else:
            input("\nInvalid choice, press any key to continue.")
    return json_settings_dictionary


#The "main_menu()" function will run a "while True"
#loop that will allow the user to navigate the menu, and the
#loop will be broken out of when they select the "Quit" option,
#or when they press Ctrl+C (SIGINT, Signal Interrupt).
def main_menu(json_settings_dictionary, json_default_settings_dictionary, cwd, json_settings_file_path_name):

    while True:
        #The "clear_screen()" function will clear the CLI screen
        #using the appropriate command depending on the operating system.
        clear_screen()

        menu_actions_dict = {
        "1": ["Generate PDF with Current Settings", generate_pdf_file, (json_settings_dictionary, json_default_settings_dictionary, cwd)],
        "2": ["Page Management Menu", page_management_menu, (json_settings_dictionary, json_default_settings_dictionary, json_settings_file_path_name)],
        "3": ["Set Maximum PDF File Size", set_max_file_size, (json_settings_dictionary, json_default_settings_dictionary, json_settings_file_path_name)],
        "4": ["Set Color Mode", set_color_mode, (json_settings_dictionary, json_default_settings_dictionary, json_settings_file_path_name)], 
        "5": ["Set DPI", set_dpi, (json_settings_dictionary, json_default_settings_dictionary, json_settings_file_path_name)], 
        "6": ["Brightness Menu", brightness_levels_menu, (json_settings_dictionary, json_default_settings_dictionary, json_settings_file_path_name)],
        "7": ["Contrast Menu", contrast_levels_menu, (json_settings_dictionary, json_default_settings_dictionary, json_settings_file_path_name)],
        "8": ["Filters Menu", filter_settings_menu, (json_settings_dictionary, json_default_settings_dictionary, json_settings_file_path_name)],
        "9": ["Auto-Cropping Menu", auto_crop_settings_menu, (json_settings_dictionary, json_default_settings_dictionary, json_settings_file_path_name)],
        "r": ["Reset Defaults", reset_all_settings, (json_settings_dictionary, json_default_settings_dictionary, json_settings_file_path_name)],
        "q": ["Quit", quit_function, ()]}

        #The function "textwrap_action_strings_in_menu_action_dict()", which takes in 
        #a menu action dictionary comprised of one character keys and values made up
        #of a three-member tuple (action string, function, function arguments).
        #The action strings ("value[0]") will be textwrapped and the modified
        #dictionary will be returned.
        menu_actions_dict = textwrap_action_strings_in_menu_action_dict(menu_actions_dict)

        print("  Analog eBooks")
        print("=== Main Menu ===\n\n")

        #The function "run_menu" will retrieve and call the function
        #at the appropriate choice key in the "menu_actions_dict"
        json_settings_dictionary = run_menu(menu_actions_dict, json_settings_dictionary)


#The "main()" function will initialize the path variables and the "json_settings_dictionary" and 
#"json_default_settings_dictionary" dictionaries by calling the "load_json_data()" function.
#It will then initiate the main menu loop by calling the "main_menu()" function.
def main():
    #Register the Signal Interrupt (SIGINT) handler that will
    #call the "signal_interrupt_signal_handler()" function 
    #when the user presses on CTRL + C to exit the app.

    #The function "signal_interrupt_signal_handler()" will call
    #"sys.exit(0)" to exit the program normally.
    signal.signal(signal.SIGINT, signal_interrupt_signal_handler)

    cwd = os.getcwd()

    if not os.path.exists(os.path.join(cwd, "Final Book PDF Files")):
        os.mkdir(os.path.join(cwd, "Final Book PDF Files"))

    #The function "get_terminal_dimensions()" will return the number of columns 
    #and rows in the console, to allow to properly format the text and dividers.
    columns, lines = get_terminal_dimensions()

    #If either the "Original Book PDF File" subfolder is missing, or if it is empty,
    #it will be created and the code will exit the application while printing the 
    #"missing_pdf_string" on-screen.
    missing_pdf_string = "\n" + textwrap.fill("Please add the scanned book's PDF file in the 'Original Book PDF File' subfolder of the Analog eBooks folder and launch the application again.", width=columns) + "\n"     
    if not os.path.exists(os.path.join(cwd, "Original Book PDF File")):
        os.mkdir(os.path.join(cwd, "Original Book PDF File"))
        sys.exit(missing_pdf_string)
    else:
        pdf_path = os.path.join(cwd, "Original Book PDF File", "*.pdf")
        pdf_files = glob.glob(pdf_path)
        if pdf_files == []:
            sys.exit(missing_pdf_string)

    json_settings_file_path_name = os.path.join(cwd, "settings.json")

    #The function "load_json_data()" will load the JSON data from file
    #and store them in the "json_settings_dictionary", or initialize the
    #dictionary based on the values of "json_default_settings_dictionary".
    json_default_settings_dictionary, json_settings_dictionary = load_json_data(json_settings_file_path_name)

    #The "main_menu()" function will run a "while True"
    #loop that will allow the user to navigate the menu, and the
    #loop will be broken out of when they select the "Quit" option,
    #or when they press Ctrl+C (SIGINT, Signal Interrupt).
    main_menu(json_settings_dictionary, json_default_settings_dictionary, cwd, json_settings_file_path_name)


if __name__ == '__main__':

    try:
        #The strings below will be used as comments in the JSON file
        #and in the menus, so they are instantiated as global variables.
        first_page_comment_stirng = "The 'First Page' is the first page from the original PDF document that is included in your final PDF document (default setting: 1)."

        last_page_comment_stirng = "The 'Last Page' is the last page from the original PDF document that is included in your final PDF document (default setting: 0, meaning until the end of the document)."

        removed_pages_comment_string = "The 'Removed Pages' setting is a list of comma-separated individual page numbers from the original PDF file that are to be removed from your final PDF file. You can also specify spans delimited by hyphens (e.g., 1, 3, 5-10). The default setting is zero, which means that no pages will be removed (default setting: 0)."        

        cover_page_mode_comment_string = "The 'Cover Page' setting will automatically generate a cover page by extracting the book title and author from your original PDF file name. Simply add a three-hyphen separator between the book title and the subtitle and/or author information, and you may also add carriage returns by including sequences of two spaces (or four spaces for two carriage returns), as in the following example: 'Book Title --- Subtitle    by  Author Name.pdf'."

        cover_page_line_spacing_comment_string = "The 'Cover Page Line Spacing' setting will set the cover page's line spacing, with a setting between 0.80 and 1.10 being recommended (default setting: 0.90)."

        cover_page_color_selection_comment_string = "The 'Cover Page Color' will set the light background color for the upper half of the cover page and the font color for the subtitle and/or author information text, with the other cover page contents being in black color. Either enter your chosen color as a Red, Green, Blue (RGB) value (three comma-separated numbers between 0 and 255, where a value of 255 represents the Red, Green or Blue channel at full intensity, e.g., '0, 255, 255' for Cyan) or as a hex code (e.g., '#00FFFF' for Cyan), or select one of the following color options (default setting: White)."

        dpi_setting_comment_string = "The 'DPI Setting' sets the resolution, in dots per inch (DPI), of the images extracted from the original PDF document (default setting: 300 DPI)."

        max_mb_per_pdf_file_comment_string = "The 'Maximal File Size' setting will set the size threshold, in megabytes (MB), at which a new output PDF file will be generated (e.g., 'Book File (Part 2).pdf'). The code keeps track of the estimated file size as it processes every page of the original PDF document. However, this estimation does not factor in optimization steps that lead to size reductions when outputting the final PDF file. You may need to specify a slightly larger threshold than the actual size of the generated PDF files. Should you want a single file to be generated, then enter a large number of MB, like the default value of 100 MB (default setting: 100.0 MB)."

        grayscale_mode_enabled_comment_string = "The 'Grayscale Mode' is for outputting PDF files in grayscale pixels, allowing for anti-aliasing (light outline around the letters that gives the text a smoother look), while the 'Black and White Mode' outputs PDF files in black and white pixels only, leading to smaller file sizes (default setting: True)."

        auto_cropping_comment_string = "The 'Auto-Cropping' mode automatically crops the pages to remove extra margins (default setting: True)."

        auto_padding_mode_comment_string = "The 'Auto-Padding' mode automatically pads the cropped pages when the 'Auto-Cropping' mode is also enabled. This will result in a uniform page size throughout your final PDF document, which makes it easier to read the PDF document with the built-in PDF readers of e-reader devices without the applications needing to manually rescale each page. The final dimensions of the pages will be set to the widest and tallest of your cropped pages (default setting: True)."

        horizontal_crop_kernel_size_height_percent_comment_string = "The kernel is comprised of a one-dimensional array that will traverse a pixel density map array along the image width that represents the count of black pixels for each column of pixels in the page image. The kernel will 'see' black pixels and will blur the text into solid chunks (convolution step) to make it easier to detect the margins of the page. The kernel size will impact how well white space gaps are allowed within a block of text. A larger kernel size allows for more gaps within the block of text (e.g., the spaces between individual letters), but may also include more artifacts (e.g., specks of ink splatter). Setting its size is a balancing act, with 2% of the initial page height giving reasonable results. You may need to increase the 'Left-Right Kernel Size' setting if you see that the pages are cropped too aggressively (default setting: 2.0%)."

        horizontal_crop_kernel_radius_kernel_size_percent_comment_string = "The kernel radius, expressed as a percentage of the kernel size, determines what overlap of black pixels within the kernel is required in order for them to be blurred together in the convolution step. The maximum value for this is the kernel size itself (complete overlap, or 100% kernel size), which wouldn't allow for any white pixels (gaps). A value of around 30% the kernel size is usually good for detecting contiguous columns of black pixels (detecting the left and right edges of the text). You may need to decrease the value of the kernel radius from its initial value of 30% to allow for more white pixel gaps in the convolution step, which would then crop less aggressively (default setting: 30.0%)."

        horizontal_crop_margin_buffer_width_percentage_comment_string = "The initial crop selection will be expanded horizontally on the left and right by an amount of pixels equal to a certain percentage of the initial image width to help avoid accidentally cropping out text (default setting: 1.5%)."

        vertical_crop_kernel_size_height_percent_comment_string = "A larger kernel size of 8% of the initial page height is used along the 'y' axis (detecting the top and bottom margins), as there could be larger white space gaps between paragraphs. Should you need to cover greater vertical gaps when detecting the top and bottom edges of the page, you may need to increase the 'Top-Bottom Kernel Size' setting from its initial value of 8% of the initial height of the page, and potentially also decrease the 'Top-Bottom Kernel Radius' setting from its value of 20% of the adjusted kernel size (default setting: 8.0%)"

        vertical_crop_kernel_radius_kernel_size_percent_comment_string = "A smaller kernel radius value of around 20% of the kernel size is used when detecting contiguous rows of black pixels (detecting the top and bottom edges of the text). A smaller threshold is used because there may be empty lines in-between paragraphs, or larger vertical spaces between the end of a chapter and the beginning of the next chapter. Should you need to cover greater vertical gaps when detecting the top and bottom edges of the page, you may need to increase the kernel size from its initial value of 8% of the initial height of the page, and potentially also decrease the kernel threshold from its value of 20% of the adjusted kernel size (default setting: 20.0%)."

        vertical_crop_margin_buffer_height_percentage_comment_string = "The initial crop selection will be expanded vertically above and below by an amount of pixels equal to a certain percentage of the initial image height to help avoid accidentally cropping out text (default setting: 2.0%)."

        initial_brightness_level_comment_string = "The 'Initial Brightness Level' setting will brighten all pixels that are not pure black in the page images extracted from the original PDF document. A value of one gives the original image (no changes in brightness), a value below one and above zero (e.g., 0.42) will decrease the brightness, while values above one will increase the brightness (default setting: 1.0, or no changes)." 

        final_brightness_level_comment_string = "The 'Final Brightness Level' setting will selectively darken the interior of the characters on the page, while minimally affecting the anti-aliasing pixels (pale outline of the characters that gives the text a smoother look). Enter a value greater than one should you like to darken the letters even more, provided that they are not already black in color (default setting: 1.0, or regular darkening of the letters)." 

        initial_contrast_level_comment_string = "The 'Initial Contrast Level' setting will adjust the contrast level of the page images extracted from the original PDF document, with a contrast level of one resulting in no changes, a value less than one and greater than zero decreasing the contrast, and a value above one increasing the contrast. Increasing the contrast will darken the colors that are darker than the initial mean darkness of all of the pixels on the page after brightening (baseline mean page color). When filtering out the paper color pixels, some slightly darker pixels will be left behind around the text. Increasing the contrast will make it easier to filter out these pixels that are only slightly darker than the baseline color (default setting: 1.0, or no changes)."

        final_contrast_level_comment_string = "The 'Final Contrast Level' adjusts the contrast one last time, once all of the filter and 'Final Brightness' changes have been applied (default setting: 1.0, or no changes)."

        dark_mode_comment_string = "The 'Dark Mode' will make the page dark-colored and the text light-colored (default setting: False)."

        left_margin_width_percent_comment_string = "The 'Margins Filter Left Margin' setting will be used if the 'Margins Filter' is ON and will specify the width of the left margin that will be submitted to the 'Margins Filter', in terms of a percentage of the initial page width (e.g., 2.5 for 2.5% of the initial page width; default setting: 2.5%)."

        right_margin_width_percent_comment_string = "The 'Margins Filter Right Margin' setting will be used if the 'Margins Filter' is ON and will specify the width of the right margin that will be submitted to the 'Margins Filter', in terms of a percentage of the initial page width (e.g., 2.5 for 2.5% of the initial page width; default setting: 2.5%)."

        top_margin_height_percent_comment_string = "The 'Margins Filter Top Margin' setting will be used if the 'Margins Filter' is ON and will specify the height of the top margin that will be submitted to the 'Margins Filter', in terms of a percentage of the initial page height (e.g., 2.5 for 2.5% of the initial page height; default setting: 2.5%)."

        bottom_margin_height_percent_comment_string = "The 'Margins Filter Bottom Margin' setting will be used if the 'Margins Filter' is ON and will specify the height of the bottom margin that will be submitted to the 'Margins Filter', in terms of a percentage of the initial page height (e.g., 2.5 for 2.5% of the initial page height; default setting: 2.5%)."

        do_filter_out_splotches_margins_comment_string = "The 'Margins Filter' setting will be used when cropping pages and will filter out grayscale pixels found in the margins that are lighter than the mean non-white pixel value on the center of the page, plus the product of the standard deviation of the non-white pixels by the value of 'Margins Filter Multiplier' (i.e., mean + 'Margins Filter Multiplier' * standard deviation), where a normal distribution of non-white pixel values is assumed (default setting: True)."

        do_filter_out_splotches_entire_page_comment_string = "The 'Full-Page Filter' setting will filter out grayscale pixels that are lighter than the mean non-white pixel value on the center of the page, plus the product of the standard deviation of the non-white pixels by the value of 'Full-Page Filter Multiplier' (i.e., mean + 'Full-Page Filter Multiplier' * standard deviation), where a normal distribution of non-white pixel values is assumed (default setting: True)."

        number_of_standard_deviations_for_filtering_page_color_comment_string = "The 'Page Color Filter Multiplier' will determine the number of standard deviations (the number may be positive or negative, and may contain decimals) that will be added to the initial mean value of all pixels on the page when filtering out pixel values greater (lighter) than: mean + 'Page Color Filter Multiplier' * standard deviation, assuming a normal distribution of pixel values (0.0 being black and 1.0 being white), where the pixel values are distributed within 3 standard deviations on either side of the mean. A value of 'Page Color Filter Multiplier' of zero will give the mean as a threshold, while positive values up to +3.0 will keep more and more original pixels, and negative values -3.0 and over will filter out pixels more aggressively (default setting: 0.0)."

        number_of_standard_deviations_for_filtering_page_color_when_cropping_comment_string = "The 'Page Color Filter Multiplier When Cropping' is only used when cropping the pages and will not affect the appearance of the text. It will determine the number of standard deviations (the number may be positive or negative, and may contain decimals) that will be added to the initial mean value of all pixels on the page when filtering out pixel values greater (lighter) than: mean + 'Page Color Filter Multiplier When Cropping' * standard deviation, assuming a normal distribution of pixel values (0.0 being black and 1.0 being white), where the pixel values are distributed within 3 standard deviations on either side of the mean. A value of 'Page Color Filter Multiplier' of zero will give the mean as a threshold, while positive values up to +3.0 will keep more and more original pixels, and negative values -3.0 and over will filter out pixels more aggressively. In this case, as the text must be blemish-free when cropping it, a more aggressive value of -1.5 is used (default setting: -1.5)."

        number_of_standard_deviations_for_filtering_splotches_margins_comment_string =  "The 'Margins Filter Multiplier' will determine the number of standard deviations (the number may be positive or negative, and may contain decimals) that will be added to the mean non-white pixel value on the center of the page when filtering out pixel values in the margins greater (lighter) than: mean + 'Margins Filter Multiplier' * standard deviation, assuming a normal distribution of non-white pixel values (0.0 being black and 1.0 being white), where the pixel values are distributed within 3 standard deviations on either side of the mean. A value of 'Margins Filter Multiplier' of zero will give the mean as a threshold, while positive values up to +3.0 will keep more and more original pixels, and negative values -3.0 and over will filter out pixels more aggressively (default setting: -0.25)."

        number_of_standard_deviations_for_filtering_splotches_entire_page_comment_string = "The 'Full-Page Filter Multiplier' will determine the number of standard deviations (the number may be positive or negative, and may contain decimals) that will be added to the mean non-white pixel value on the center of the page when filtering out pixel values greater (lighter) than: mean + 'Full-Page Filter Multiplier' * standard deviation, assuming a normal distribution of non-white pixel values (0.0 being black and 1.0 being white), where the pixel values are distributed within 3 standard deviations on either side of the mean. A value of 'Full-Page Filter Multiplier' of zero will give the mean as a threshold, while positive values up to +3.0 will keep more and more original pixels, and negative values -3.0 and over will filter out pixels more aggressively (default setting: 3.0)."

        colors_dict = {
            (255, 255, 255) : "White",
            #Creams and Yellows
            (255, 215, 0) : "Gold", #Contrast Ratio 14.97:1 (WCAG AAA Pass Normal and Large Text) on webaim.org/resources/contrastchecker
            (255, 236, 122) : "Corn Yellow", #Contrast Ratio 17.53:1 (WCAG AAA Pass Normal and Large Text) on webaim.org/resources/contrastchecker
            #Coral Pinks
            (240, 128, 128) : "Light Coral", #Contrast Ratio 8.1:1 (WCAG AAA Pass Normal and Large Text) on webaim.org/resources/contrastchecker
            (255, 160, 122) : "Light Salmon", #Contrast Ratio 10.56:1 (WCAG AAA Pass Normal and Large Text) on webaim.org/resources/contrastchecker
            #Light Greens and Blues
            (144, 238, 144) : "Light Green", #Contrast Ratio 14.81:1 (WCAG AAA Pass Normal and Large Text) on webaim.org/resources/contrastchecker
            (127, 255, 212) : "Aquamarine", #Contrast Ratio 17.15:1 (WCAG AAA Pass Normal and Large Text) on webaim.org/resources/contrastchecker
            (0, 206, 209) : "Dark Turquoise", #Contrast Ratio 10.74:1 (WCAG AAA Pass Normal and Large Text) on webaim.org/resources/contrastchecker
            (135, 206, 250) : "Light Sky Blue" #Contrast Ratio 12.23:1 (WCAG AAA Pass Normal and Large Text) on webaim.org/resources/contrastchecker      
        }

        main()

    except Exception as e:
        #The function "get_terminal_dimensions()" will return the number of columns 
        #and rows in the console, to allow to properly format the text and dividers.
        columns, lines = get_terminal_dimensions()
        troubleshooting_step_1_string = textwrap.fill("1. Please manually back up 'settings.json' if you need to salvage your user settings.", width=columns)
        troubleshooting_step_2_string = textwrap.fill("2. Once backed up, you can delete the original copy of 'settings.json' in the root folder to reset to the default settings and launch the app again.", width=columns)

        print("\n" + "=" * columns)
        print("CRITICAL ERRROR ENCOUNTERED")
        print("\nDetails:", e)
        print("\n" + "=" * columns)

        print("\nTroubleshooting Steps:\n")
        print(troubleshooting_step_1_string)
        print(troubleshooting_step_2_string)

        #The function "write_entry_in_error_log()" will write 
        #the full technical traceback error to the error log.
        write_entry_in_error_log()

        #Exit with error code
        sys.exit(1)



