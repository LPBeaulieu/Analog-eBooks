# Analog eBooks
This CLI application removes the page color and crops the text from scanned PDF books so that you could read them on your e-reader!

![Analog eBooks Demo](https://github.com/LPBeaulieu/Analog-eBooks/blob/main/Analog%20eBooks%20Demo.png)
<h3 align="center">Analog eBooks</h3>
<div align="center">
  
  [![License: AGPL-3.0](https://img.shields.io/badge/License-AGPLv3.0-brightgreen.svg)](https://github.com/LPBeaulieu/Analog-eBooks/blob/main/LICENSE)
  
</div>

## 📝 Table of Contents
- [Getting Started](#getting_started)
- [Usage](#usage)
- [Author](#author)
- [Acknowledgments](#acknowledgments)

## 🏁 Getting Started <a name = "getting_started"></a>
Analog eBooks is a Command-Line Interface (CLI) application that allows to process high-quality
book scans (pages being already deskewed and cropped to remove most of the shadows on
the edges of the pages). It is most useful when generating simple grayscale PDF
documents that can readily be read on e-readers.

To find such high-quality book scans, head over to https://archive.org/details/internetarchivebooks 
and use the following targeted query in your Internet Archive search (please consider donating to 
Internet Archive to help their archiving efforts!):
```
date:([* TO 1930-12-31]) AND format:(PDF) AND -collection:(community OR opensource) AND (title:("Book Title") OR description:("Book Title")) AND creator:("Author Name")
```
  - 'date:([* TO 1930-12-31])': Will only include results that are now in the public domain (adjust the date accordingly, which increases
    the chance of you being able to download a PDF (I do not condone pirating books, so please limit yourself to public domain works).
  - 'format:(PDF)': Will only include results that have a downloadable PDF version.
  - '-collection:(community OR opensource)': to remove most user-uploaded files that lack professional 
      post-processing, where the minus sign ("-") indicates to exclude these results.
  - '(title:("Book Title") OR description:("Book Title")) AND creator:("Author Name")': Any book titles and 
    author names that contain spaces should be placed within quotes (" ").
  - Instead of '-collection:(community OR opensource)', other alternative options for filters are:
    - 'scanner:(scribe*)': Will only retain high-quality book scans that have been deskewed and cropped.
    - 'collection:(americana OR toronto OR canadiana OR universallibrary OR europeanlibraries OR library_of_congress OR university_of_california_libraries OR harvard OR oxford_university)': 
      Books from collections known for high standards.
    - 'lccn:*': Library of Congress Control Number: Searching for books that have a LCCN ensures 
      that they have been professionally catalogued.
    - 'oclc-id:*': Books linked to the OCLC Worldcat database
    - 'sponsor:("Sloan Foundation")': This group funded the digitization 
      of a large quantity of high-quality public domain books.
    - 'sponsor:("Library and Archives Canada" OR "Better World Books")' Groups for preserving
      out-of-copyright books or books donated from library systems, respectively 
    - 'operator:"scanner-*"' or 'operator:"associate-*': Items processed by professional Internet Archive staff.
    - 'republisher_operator:*': This identifies the specific technician who performed the cropping/deskewing 
      post-processing, indicating that the book was likely processed according to the standard Internet Archive
      quality-control guidelines.
    - If you find a scanned book on Internet Archive that you deem particularly well scanned, 
      scroll to the bottom of its page and click "All Files". Open the '_meta.xml' file. Any
      field you see in that XML list (e.g., curation) may be used as a filter in the Advanced
      Search bar to find other books processed in the exact same batch or facility.

To use Analog eBooks, simply unzip the zipped folder from the ['Releases'](https://github.com/LPBeaulieu/Analog-eBooks/releases/) section in a location 
where you have permissions access, such as 'Documents', and NOT 'Windows' nor 'Program Files', 
as the code will need to have writing access to generate PDFs and modify the JSON file holding 
all of your settings. There should be an 'Original Book PDF File' subfolder (in which you will 
place the original scanned book PDF file) and a 'Final Book PDF Files' subfolder (where Analog 
eBooks will generate the PDF documents) within the unzipped folder. Simply place a scanned book 
pdf in the 'Original Book PDF File' folder, double-click on 'Analog eBooks.exe' you're ready to go!

Should you instead wish to run the source code file ('Analog eBooks.py'), you would need to 
install its dependencies, namely NumPy and PyMuPDF:
```
py -m pip install numpy pymupdf
```
You would then run the code as follows:
```
py "Analog eBooks.py"
```

## 🎈 Usage <a name="usage"></a>  
The default settings should work well for most books, where you will only need to specify the first 
and last pages in the 'Page Management' menu (main menu item 2). I would recommend skipping over all 
of the front matter and starting at the first page of the body text, as pages that don't have much text 
on them aren't optimally dealt with by the code. Select the very last page of body text as the 'Last Page',
in order to avoid including blank pages and the back cover image. The code will print out a list of what 
could be blank pages after processing the original PDF file. To remove these pages, you may copy and paste 
that list (make sure to remove any carriage returns, if present, by first pasting it in Notepad) for use 
in the 'Removed Pages' submenu (submenu item 3) from the 'Page Management' menu, before running the code 
again (main menu item 1). You may also wish to remove full-page illustrations, as these tend to be poorly 
cropped and do not typically render very well. Should the left and right margins be too wide to afford 
nicely-sized text on your e-reader, it is likely because some pages weren't properly cropped. You may 
turn the 'Auto-Padding' mode off in the main menu item 9 ('Auto-Cropping Menu') and generate the 
PDF file once again in order to more easily locate the problematic page. You could then either 
pre-crop that page in the original PDF file with a PDF editing software like PDFgear (more on that below), 
or simply read the unpadded PDF document on your e-reader with KOReader, where padding isn't all that 
important. A cover page will be generated by default by the code (see the 'Cover Page' setting detailed below). 

A series of settings allow you to fine-tune the results, which will be summarized below
(I think you should at least read the three first entries: 'Page Management Menu', 
'Cover Page' and 'Maximum PDF File Size', as these are likely the only settings
that you will need to adjust in most cases).

- Page Management Menu: This menu will let you set the first and last pages from the original PDF
  that will make their way into the final PDF document, provided that you didn't include these in
  the list of removed pages (e.g. 1-4, 6, 330-335). The settings for the cover page are also found
  in this menu (see below).

- Cover Page: A cover page is automatically generated by default by extracting the book title and
  author information from the original PDF file name. Simply include a three-hyphen separator between
  the book title and the subtitle and/or author information, and you may also add carriage returns 
  by including sequences of two or more spaces, as in the following example: 
  'Book Title --- Subtitle    by  Author Name.pdf'
  Note that four consecutive spaces will result in an empty line between the "Subtitle" and "by"
  on the cover page, which looks nice on the finished cover page.
  
  You may also include a TTF or OTF font file in the application's root folder in order to
  use a custom font for the cover page. You may need to alter the default line spacing of
  0.9 to better suit your font.

  You may choose one of 8 color options instead of the default white color for the cover page,
  or specify your color of choice as an RGB value (e.g., '0, 255, 255' for Cyan) or a hex code
  (e.g., '#00FFFF' for Cyan).

  The cover page line spacing and color may be set in the "Cover Page" submenu of the 'Page Management' menu.

- Maximum PDF File Size: The code will set the size threshold, in megabytes (MB), at which a new 
  output PDF file will be generated (e.g., 'Book File (Part 2).pdf'). The code keeps track of the 
  estimated file size as it processes every page of the original PDF document. However, this estimation 
  does not factor in optimization steps that lead to size reductions when outputting the final PDF file. 
  You may need to specify a slightly larger threshold than the actual size of the generated PDF files. 
  Should you want a single file to be generated, then enter a large number of MB, like the default value 
  of 100 MB.

- Color Mode: The PDF documents may be outputted either in 'Grayscale Mode' or 'Black and White Mode',
  with the 'Grayscale Mode' being the default and preferred mode to generate text with anti-aliasing (dark 
  characters with a pale outline that gives the text a smoother look). The 'Black and White Mode' generates
  PDF documents in black and white pixels only, which leads to smaller file sizes.

- Dark Mode: This mode inverts the grayscale colors of the generated PDF documents, where the text
  is light-colored and the pages are dark-colored.

- DPI: The DPI setting will set the resolution, in dots per inch (DPI) at which the images will be
  extracted from the original PDF file. Aim for 200-300 dpi for best results, with 300 dpi giving
  larger PDF files.

- Initial Brightness: The 'Initial Brightness Level' setting will brighten all pixels that are not pure 
  black in the page images extracted from the original PDF document. A value of one gives the original 
  image (no changes in brightness), a value below one and above zero (e.g., 0.42) will decrease the brightness, 
  while values above one will increase the brightness (default setting: 1.0, or no changes).

- Final Brightness: The 'Final Brightness Level' setting will selectively darken the 
  inside of the characters on the page, while minimally affecting the anti-aliasing 
  pixels (pale outline of the characters that gives the text a smoother look). 
  Enter a value greater than one should you like to darken the letters 
  even more, provided that they are not already black in color 
  (default setting: 1.0, or regular darkening of the letters). 

- Initial Contrast: The 'Initial Contrast Level' setting will adjust the contrast level of the 
  page images extracted from the original PDF document, with a contrast level of one resulting 
  in no changes, a value less than one and greater than zero decreasing the contrast, and a 
  value above one increasing the contrast. Increasing the contrast will darken the colors 
  that are darker than the initial mean darkness of all of the pixels on the page after 
  brightening (baseline mean page color). When filtering out the paper color pixels, 
  some slightly darker pixels will be left behind around the text. Increasing the 
  contrast will make it easier to filter out these pixels that are only slightly 
  darker than the baseline color, as these will not be significantly darkened 
  in the contrast step, while the text will become much darker 
  (default setting: 1.0, or no changes).

- Final Contrast: The 'Final Contrast Level' adjusts the contrast one last time, 
  once all of the filter and 'Final Brightness' changes have been applied.
  (default setting: 1.0, or no changes).

- Filters: There are three stages where pixels are filtered out in order to remove the background color or
  blemishes on the pages, which are applied in sequence. You may need to review the basics of how a normal
  distribution works (the values are distributed equally around the mean in a bell curve, with virtually all
  data points being found within 3.0 standard deviations on either side of the mean. Having a basic grasp of 
  what a normal distribution looks like will help you evaluate where to place the threshold for these filters.
  
  1- Page Color Filter: This filter removes pixels that are lighter than the mean grayscale value of all
     pixels found on the page image extracted from the original PDF file, plus a certain multiplier times
     the standard deviation of the grayscale pixel value (threshold = mean + multiplier * standard deviation). 
     Tuning the value of the multiplier will adjust the threshold above which the pixels will be converted to
     white pixels. If the multiplier equals zero, then the threshold equals the mean pixel grayscale value on
     the page. If the multiplier is greater than zero, then it means that less pixels will be filtered out, up
     to a value of about +3.0, where the filter will leave the page more or less unchanged. The lower the value
     of the multiplier, the more pixels will be filtered out. It is generally fine to use a rather stringent 
     threshold equal to the mean, with a multiplier of 0.0 (mean + 0.0 * standard deviation) at this stage 
     though, as most pixels on the page are of the page color (much lighter) and not of the text color (much darker). 

     Page Color Filter When Cropping: This filter is only used when cropping the pages and will not affect the
     appearance of the text. It operates on the same principle as the 'Page Color Filter', but it removes more 
     non-white pixels so as to ensure that the blotches are completely removed, thus making it easier to crop 
     the text. The default value of the multiplier for this filter is then lower than that of the regular 
     'Page Color Filter' (-1.5 vs 0.0).

  2- Margins Filter: This filter is only used when cropping the pages and applies to the margins of the 
     page (you can set the width of the zones that will be subjected to the margins filter by setting the 
     'Margins Filter Left Margin', the 'Margins Filter Right Margin', the 'Margins Filter Top Margin' and the 
     'Margins Filter Bottom Margin' values. This filter filters out pixels that are lighter than the mean 
     grayscale value of the non-white pixels in the center of the page (outside of the abovementioned margins) 
     plus a certain multiplier times the standard deviation of the grayscale pixel value (threshold = mean + 
     multiplier * standard deviation). Despite the distribution of pixels now only being the non-white pixels 
     in the center of the page (meaning that the mean grayscale pixel value is much closer to black (0.0) than 
     white (1.0), as opposed to the Page Color Filter that had a distribution of ALL pixels in the page and of 
     which the mean value was much closer to white, since most of the page was comprised of the paper's color), 
     you can still use a relatively stringent value around -0.25 for the multiplier, as there shouldn't be much 
     text in the margins. This will allow you to filter out shadows cast by the spine and the page yellowing that 
     tends to be more pronounced around the outer edges of the pages.

  3- Full-Page Filter: This filter applies to the full page, but unlike the Page Color filter, the distribution
     of pixels ONLY includes non-white pixels in the center of the page (meaning that the mean grayscale pixel 
     value is much closer to black (0.0) than white (1.0), as opposed to the Page Color Filter that had a 
     distribution of ALL pixels in the page and of which the mean value was much closer to white, since most 
     of the initial page image was comprised of the paper's color, and the text's pixels were then at the very 
     left end of the bell curve.  
     
     This means that you need to be more conservative when filtering out pixels with the Full-Page Filter, 
     as you want to keep the text pixels in. By default, a multiplier of +3.0 is used, meaning that virtually
     all of the original pixels are maintained in this step, as the first 'Page Color Filter' usually does a
     good job at removing the page color. Should there be numerous remaining blotches on most pages, you might
     need to decrease the value of this multiplier from its default value of +3.0, and potentially also
     increase the 'Initial Contrast Level' setting, which would darken the text much more than the
     light blotches on the pages.

- Auto-Cropping: When the Auto-Cropping mode is enabled (default behaviour), the code will
  automatically crop the left, right, top and bottom margins. To do this, it first converts
  the image to a black and white visualization of the pixels, where all non-white pixels
  are represented by the value 1 and the white pixels are represented by the value 0. 
  When trying to find the left and right margins, all of the rows of pixels for each 
  column are added together to form a pixel density map (a single row containing the
  sum of all rows of non-white pixels (1s) in each column of the image. We can then 
  determine where are the left and right edges of the image, as the sums will be 
  greatest where the block of text is. However, as there are white spaces between letters 
  and indentations, it is best to "smudge" the letters together by a process called 
  convolution.

  This will produce contiguous "chunks" of darkness that will make detecting the edges
  much easier. The convolution step uses a window (like a magnifying lens of sorts) 
  called the "kernel". It will traverse the row of summed-up 1s and join up portions
  of the row into chunks if they fall below a threshold of maximal allowed gaps 
  (0s or white space) between areas where the sum is above zero. If the magnifying
  glass is larger, then it can "connect" larger gaps in the text. That amounts to
  increasing the kernel size. The threshold is calculated by multiplying the kernel
  size by the kernel radius, expressed as a percentage of the kernel size. The kernel
  radius represents the maximal acceptable white gaps in-between non-white pixels, 
  where a kernel radius of 100% of the kernel size allows for no white gaps. 

  The left-right edge detection and top-bottom edge detection each use their own
  kernel size and kernel radius settings, as there are far larger white space gaps
  vertically (e.g., the space below a chapter heading) than horizontally (spaces between
  letters), so the kernel size needs to be greater vertically and the kernel radius must
  be only about 20% of the kernel size vertically (vs 30% horizontally), in order to allow 
  for more white space gaps when detecting the top-bottom edges of the block of text. 

  Should there be rather stark shadows cast by the book's spine, these might get darkened
  in the contrast steps, as they are darker than the average page color. This means that 
  they are difficult to filter out and you might need to preprocess the original pages
  that have such shadows in the original PDF file with a PDF editing software such as
  'PDFgear' (https://www.pdfgear.com/pdfgear-for-windows/). Use the desktop version that 
  allows you to edit large PDF documents. You could then manually crop out the left or 
  right edges of the pages that have such shadows (those that weren't cropped properly
  by Analog eBooks), save the edited PDF document and run Analog eBooks on it for better 
  results, particularly when using the 'Auto-Padding' mode (see below), where the widest 
  and tallest dimensions of all of your pages will be used to determine the size of the 
  pages in your final PDF.
  
- Auto-Padding: The 'Auto-Padding' mode automatically pads the cropped pages when the 
  'Auto-Cropping' mode is also enabled. This will result in a uniform page size throughout
  your final PDF document, which makes it easier to read the PDF document with the built-in 
  PDF readers of e-reader devices without the applications needing to manually rescale each 
  page, which can lead to crashes. The final dimensions of the pages will be set to the widest 
  and tallest of your cropped pages (see comment about the cropping step above). However, you 
  can reliably read unpadded pages (your pages all slightly differing in size) if you use the 
  KOReader app (more on that below). 

- Use KOReader to read the grayscale PDF documents generated with Analog eBooks 
  for best results. KOReader is a third-party, open-source e-reader application that 
  reads large PDF documents seamlessly on e-reader devices. It allows to display the 
  pages very quickly even though the document doesn't have uniform page sizing. 
  My recommendation is to turn the Analog eReader's Auto-Cropping mode ON and the
  'Auto-Padding' mode OFF when using KOReader in order to maximize the text size, 
  especially when reading in landscape mode on 6-inch e-readers, where screen 
  real estate is at a premium. You may adjust both these settings in the main 
  menu item 9 ('Auto-Cropping Menu'). KOReader also lets you adjust the 
  contrast level of the text dynamically (on a page by page level), 
  so you may adjust the weight of the text just the way you like it.

  As the PDF documents generated with Analog eBooks do not have OCR data,
  you will get pesky KOReader notifications should you hold your finger slightly
  too long while turning pages. To disable these, follow the steps below:

    How to remove the OCR notifications in KOReader:
    - Go to the 'Settings' menu by selecting the gear icon at the top of the screen.
    - Select 'Taps and Gestures', 'Long-Press on text' and select the radio button 
      'Do nothing'.

  Relevant KOReader links:
    - https://koreader.rocks/
    - https://github.com/koreader/koreader/wiki/Installation-on-Kobo-devices

- You may change all of these settings through the Command-Line Interface (CLI) menu system, 
  or by directly modifying the 'settings.json' JSON file that is located in he application's 
  root folder. To do this, open the JSON file in Notepad and make sure that the 'Word Wrap' 
  setting is turned on in the 'View' menu to properly display the comments that explain all 
  of the settings. A few notes of caution when directly modifying the JSON file:
    - You must always use double quotes when setting a string-type value, such
      as the list of removed pages: (e.g., use this:  "1-2, 5-10", NOT this:  '1-2, 5-10').
    - Always save and then close the JSON file before running Analog eBooks.
    - Should there be any invalid values in the 'settings.json' file, the file will be 
      overwritten at runtime with a fresh copy of 'settings.json' with the default settings.
  
  A summary of your selected settings will be displayed as you are creating the PDF document,
  including:
    - Cover Page ON/OFF 
      Cover Page Line Spacing
      Cover Page Color
      Cover Page Preview
    - First Page (if not in list of deleted pages)
    - Last Page (if not in list of deleted pages)
    - List of Removed Pages
    - Max File Size (MB)
    - Color Mode (Grayscale or Black and White, Dark Mode ON/OFF)
    - Auto-Cropping ON/OFF
    - Auto-Padding ON/OFF

Happy reading!

## ✍️ Authors <a name = "author"></a>
- 👋 Hi, I’m Louis-Philippe!
- 👀 I’m interested in natural language processing (NLP) and anything to do with words, really! 📝
- 🌱 I’m currently reading about deep learning (and reviewing the underlying math involved in coding such applications 🧮😕)
- 📫 How to reach me: By e-mail! LPBeaulieu@gmail.com 💻


## 🎉 Acknowledgments <a name = "acknowledgments"></a>
- Hat tip to [@kylelobo](https://github.com/kylelobo) for the GitHub README template!


