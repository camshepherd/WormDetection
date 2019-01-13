import cv2
import os
import time

# The program is self-contained within worms.py and this file can be run straight from the python interpreter. The
# script requires that the images that it is reading appear in a particular location. The script worms.py must be
# located in the same directory as another directory called Images. The directory Images must contain another directory
# BBBC010_v1_images. Within BBBC010_v1_images there will be a number of files that come in pairs, and it is assumed
# that all images will be alphabetically ordered such that w1 and w2 files relating to the same dish will be adjacent
#  and that the w1 file will always come first.
# The script worms.py will also expect that there exists a directory in the same directory as worms.py called Testing.
#  This directory will hold images from the last cycle only of the previous dish analysed.
# Finally worms.py must have in the same directory as it another file called cleaned_images. This is where worms.py
# will save the cleaned, labelled and coloured images after processing.
# The function in worms.py process_images() will take parameters detailing the file locations of w1 and w2 files
# directly and can be invoked by the user with varying parameters so that the directory structure regarding the
# location of input and output files need not be maintained.


# hard code a relatively large number of repeating distinct colours for their later use in the user-friendly output of worms
colour_colours = [(255,0,255),(255,0,0),(0,255,255),(255,255,255),(0,0,255),(255,255,0),(0,255,0),(255,0,255),(255,0,0),(0,255,255),(255,255,255),(0,0,255),(255,255,0),(0,255,0),(255,0,255),(255,0,0),(0,255,255),(255,255,255),(0,0,255),(255,255,0),(0,255,0), (255,0,255),(255,0,0),(0,255,255),(255,255,255),(0,0,255),(255,255,0),(0,255,0), (255,0,255),(255,0,0),(0,255,255),(255,255,255),(0,0,255),(255,255,0),(0,255,0), (255,0,255),(255,0,0),(0,255,255),(255,255,255),(0,0,255),(255,255,0),(0,255,0), (255,0,255),(255,0,0),(0,255,255),(255,255,255),(0,0,255),(255,255,0),(0,255,0)]



def get_bw_image_w1(file_location):
    ''' Input: w1 file_location as string. Output: 8-bit grayscale representation '''
    image = cv2.imread(file_location, cv2.IMREAD_COLOR)
    cv2.imwrite("./testing/w1_bw_initial.tif", image)
    # import the image in full 16-bit, converting to 8-bit later
    image = cv2.normalize(image, 0, 60000, cv2.NORM_MINMAX)
    image = image.astype("uint8")
    cv2.imwrite("./testing/w1_bw_normalised.tif", image)
    image = cv2.GaussianBlur(image, (13,13),0)
    # the first parameter from cv2.threshold() is not needed, therefore ignore it
    ignore, image = cv2.threshold(image, 100, 255, cv2.THRESH_BINARY)
    cv2.imwrite("./testing/w1_bw_final.tif", image)
    return image


def get_bw_image_w2(file_location):
    ''' Input: w2 file_location as string. Output: 8-bit grayscale representation '''
    image = cv2.imread(file_location, cv2.IMREAD_ANYDEPTH)
    cv2.imwrite("./testing/w2_bw_initial.tif", image)
    # import the image in full 16-bit, only converting to 8-bit later
    image = cv2.normalize(image, 0, 65000, cv2.NORM_MINMAX)
    image = image.astype("uint8")
    # Convert down to 8-bit as the required functions for image manipulation do not accept 16-bit image channels
    cv2.imwrite("./testing/w2_bw_normalised.tif", image)
    image = cv2.GaussianBlur(image, (15,15),0)
    # run adaptive thresholding on the image and invert
    image = cv2.adaptiveThreshold(image, 255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,33,5)
    image = 255 - image[:]
    cv2.imwrite("./testing/w2_bw_final.tif", image)
    return image


def get_edge_mask(image):
    ''' Input: 8-bit image. Output: set of contours from within the image'''
    height, width = image.shape
    # copy the image as findContours() is destructive
    image_copy = image.copy()
    ignore, contours, hierarchy = cv2.findContours(image_copy, cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    sorted(contours, key=lambda contour: len(contour))
    # return the sorted set of contours so that the first one is likely to be the contour marking the edge of the subject dish
    return contours


def paint_image(input_image):
    ''' Input: 8-bit grayscale image. Output: 8-bit grayscale image with unique worms painted with unique intensities, number of unique worms found '''
    height, width = input_image.shape
    num_worms = 0
    # Ensure that every pixel in the image is either 0 or 255
    for y in range(height):
        for x in range(width):
            if input_image[y][x].any() >= 1:
                input_image[y][x] = 255
            else:
                input_image[y][x] = 0
    cv2.imwrite(".testing/paint_image_initial.tif", input_image)
    # make copies of the input_image so that floodFill() will not damage input_image
    working_image = input_image.copy()
    colour = 1
    mask = working_image.copy()
    # the mask for floodFill must be 2 pixels wider and higher than the image to be operated on
    for y in range(height):
        for x in range(width):
            mask[y][x] = 0
    mask = cv2.resize(mask, (width + 2, height + 2))

    # Paint every blob of adjacent pixels a different intensity
    for y in range(height):
        for x in range(width):
            if working_image[y][x] == 255:
                cv2.floodFill(working_image, mask, (x,y), colour)
                colour += 1
    cv2.imwrite("./testing/paint_image_post_fill.tif", working_image)
    # Look over all the painted blobs, removing any that are too small, and changing the intensities of the remaining
    # blobs so that every intensity between 1 and the maximum exists somewhere in the image
    final_colour = 255
    for colour2 in range(1, colour):
        sum = 0
        for y in range(height):
            for x in range(width):
                if working_image[y][x] == colour2:
                    sum += 1
        if sum <= 120:
            for y in range(height):
                for x in range(width):
                    if working_image[y][x] == colour2:
                        working_image[y][x] = 0
        else:
            final_colour -= 1
            num_worms += 1
            for y in range(height):
                for x in range(width):
                    if working_image[y][x] == colour2:
                        working_image[y][x] = final_colour

    cv2.imwrite("./testing/paint_image_final.tif", working_image)
    return working_image, 255 - final_colour


def label_worms(input_image, num_colours):
    ''' Input: 8-bit grayscale image pre-processed by paint_image(), number of distinct blobs/colours present in input_image. Output: 8-bit BGR-coloured image with worms labelled'''
    num_worms = 0
    height, width = input_image.shape
    output_image = input_image.copy()
    # the output image is generated based on the state of the input_image
    output_image = cv2.cvtColor(output_image, cv2.COLOR_GRAY2BGR)
    coord = ()
    # for every colour present in the image, collect all the pixels of that colour, paint them for ease of human-viewing and add a label with an estimate of the number of worms in that single-coloured section
    for colour in range(254, 254 - num_colours, -1):
        got_coord = False
        sum = 0
        for y in range(height):
            for x in range(width):
                if input_image[y][x] == colour:
                    output_image[y][x] = colour_colours[254-colour]
                    if not got_coord:
                        # always place the label at the top left pixel of the blob
                        coord = (x,y)
                        got_coord = True
                    sum += 1
        if 0 < sum <= 1000:
            num_worms += 1
            for y in range(height):
                for x in range(width):
                    if input_image[y][x] == colour:
                        output_image[y][x] = colour_colours[254-colour]
            cv2.putText(output_image, "1 Worm", coord, cv2.FONT_HERSHEY_DUPLEX, 0.5, colour_colours[254 - colour])
        elif sum > 1000:
            # estimate the number of worms in the blob based on the number of pixels involved. This is able to work effectively as worms do not tend to be overly variant in size
            quotient, remainder = divmod(sum, 800)
            num_worms += quotient
            for y in range(height):
                for x in range(width):
                    if input_image[y][x] == colour:
                        output_image[y][x] = colour_colours[254-colour]
            cv2.putText(output_image, str(quotient) + " Worms", coord, cv2.FONT_HERSHEY_DUPLEX, 0.5, colour_colours[254 - colour])
    # Add a label detailing the total number of worms estimated in the image
    cv2.putText(output_image, "Total number of worms = " + str(num_worms), (23,23), cv2.FONT_HERSHEY_COMPLEX, 1, 255)
    return output_image


def get_bw_image_accuracy(bw_image, bw_image_name):
    ''' Input: 8-bit grayscale image to be evaluated, name of the image as string. Output: none, will save to file accuracy of the images'''
    working_copy = bw_image.copy()
    height, width = bw_image.shape
    # force the image to comply properly with the expected binary standard
    for y in range(height):
        for x in range(width):
            if working_copy[y][x] > 0:
                working_copy[y][x] = 255

    #make copies of the images just in case the operations are destructive
    foreground_image = cv2.imread("./Images/BBBC010_v1_foreground/" + bw_image_name[32:36] + "binary.png", cv2.IMREAD_GRAYSCALE)
    bw_copy = working_copy.copy()
    foreground_copy = foreground_image.copy()
    bw_copy2 = working_copy.copy()
    foreground_copy2 = foreground_image.copy()
    bitwise_and = cv2.bitwise_and(foreground_copy, bw_copy)
    bitwise_or = cv2.bitwise_or(foreground_copy2, bw_copy2)
    # get number of white pixels in each of the images composite
    num_bitwise_and = 0
    num_bitwise_or = 0
    for y in range(height):
        for x in range(width):
            if bitwise_and[y][x] == 255:
                num_bitwise_and += 1
            if bitwise_or[y][x] == 255:
                num_bitwise_or += 1

    image_similarity = num_bitwise_and / num_bitwise_or
    # accuracy value = num_bitwise_and / num_bitwise_or
    with open("file_closeness.txt", 'a') as output_file:
        output_file.write("" + bw_image_name[32:35] + " accuracy: " + str(image_similarity) + "\n")




def process_images(save_location, w1, w2):
    ''' Input: Final image save_location as string, path to image taken with w1 as string, path to image taken with w2 as string. Output: None'''
    start_time = time.time()
    # Process each of w1 and w2, ready for merging
    imageA = get_bw_image_w1(w1)
    imageB = get_bw_image_w2(w2)
    # imageB must have three 8-bit channels to be bitwise-operated with imageA
    imageB = cv2.merge([imageB, imageB, imageB])
    imageSum = imageA.astype("uint8") + imageB.astype("uint8")
    imageSum = cv2.cvtColor(imageSum, cv2.COLOR_BGR2GRAY)
    cv2.imwrite("./testing/process_bw_addition_initial.tif", imageSum)
    edge_mask = get_edge_mask(imageSum)
    cv2.imwrite("./testing/process_bw_addition_masked.tif", imageSum)
    cv2.drawContours(imageSum, edge_mask, 0, 0, 25)

    cv2.imwrite("./testing/process_bw_addition_applied_contours.tif", imageSum)
    imageSum, num_colours = paint_image(imageSum)

    #empty the log file then get a value for image accuracy for each image before worm counting
    open("file_closeness.txt", 'w').close()
    get_bw_image_accuracy(imageSum, w1[28:])
    imageSum = label_worms(imageSum, num_colours)
    cv2.imshow("Final image for " + save_location, imageSum)
    cv2.imwrite("./testing/process_bw_addition_labelled.tif", imageSum)
    print("time to complete " + save_location + "= ", time.time() - start_time)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    image_directory = "./Images/BBBC010_v1_images"
    the_directory = []
    for x in os.listdir(image_directory):
        the_directory.append(x) if x.endswith(".tif") else print("non-.tif file was found in image_directory")
    for k in range(0,len(the_directory), 2):
        print("Analysing image for image: " + the_directory[k])
        process_images(the_directory[k], os.path.join(image_directory,the_directory[k]), os.path.join(image_directory,the_directory[k+1]))
