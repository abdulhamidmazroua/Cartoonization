import os
import PIL
import sys
import glob
import logging
import argparse
import numpy as np
from tqdm import tqdm
from datetime import datetime
from architecture.cartoongan import cartoongan

STYLES = ["shinkai", "hayao", "hosoda", "paprika"]
VALID_EXTENSIONS = ['jpg', 'png']

parser = argparse.ArgumentParser(description="transform real world images to specified cartoon style(s)")

# main options
parser.add_argument("--styles", nargs="+", default=[STYLES[0]],
                    help="specify (multiple) cartoon styles which will be used to transform input images.")
parser.add_argument("--all_styles", action="store_true",
                    help="set true if all styled results are desired")
parser.add_argument("--input_dir", type=str, default="input_images",
                    help="directory with images to be transformed")
parser.add_argument("--output_dir", type=str, default="output_images",
                    help="directory where transformed images are saved")
parser.add_argument("--batch_size", type=int, default=1,
                    help="number of images that will be transformed in parallel to speed up processing. "
                         "higher value like 4 is recommended if there are gpus.")
parser.add_argument("--overwrite", action="store_true",
                    help="enable this if you want to regenerate outputs regardless of existing results")
parser.add_argument("--skip_comparison", action="store_true",
                    help="enable this if you only want individual style result and to save processing time")
parser.add_argument("--comparison_view", type=str, default="smart",
                    choices=["smart", "horizontal", "vertical", "grid"],
                    help="specify how input images and transformed images are concatenated for easier comparison")

# resizing options
parser.add_argument("--keep_original_size", action="store_true",
                    help="by default the input images will be resized to reasonable size to prevent potential large "
                         "computation and to save file sizes. Enable this if you want the original image size.")
parser.add_argument("--max_resized_height", type=int, default=300,
                    help="specify the max height of a image after resizing. the resized image will have the same"
                         "aspect ratio. Set higher value or enable `keep_original_size` if you want larger image.")

# logger options
parser.add_argument("--logging_lvl", type=str, default="info",
                    choices=["debug", "info", "warning", "error", "critical"],
                    help="logging level which decide how verbosely the program will be. set to `debug` if necessary")
parser.add_argument("--debug", action="store_true",
                    help="show the most detailed logging messages for debugging purpose")
parser.add_argument("--show_tf_cpp_log", action="store_true")


args = parser.parse_args()
TEMPORARY_DIR = os.path.join(f"{args.output_dir}", ".tmp")

logger = logging.getLogger("Cartoonizer")
logger.propagate = False
log_lvl = {"debug": logging.DEBUG, "info": logging.INFO,
           "warning": logging.WARNING, "error": logging.ERROR,
           "critical": logging.CRITICAL}
if args.debug:
    logger.setLevel(logging.DEBUG)
else:
    logger.setLevel(log_lvl[args.logging_lvl])
formatter = logging.Formatter(
    "[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S")

stdhandler = logging.StreamHandler(sys.stdout)
stdhandler.setFormatter(formatter)
logger.addHandler(stdhandler)

if not args.show_tf_cpp_log:
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# preprocessing
# opens the image
# convert to rgb
# convert to numpy array
# convert values to float 32
# flattening the ndarray
def pre_processing(image_path, expand_dim=True):
    input_image = PIL.Image.open(image_path).convert("RGB")

    # used to calc new size of height and width
    if not args.keep_original_size:
        width, height = input_image.size
        aspect_ratio = width / height
        resized_height = min(height, args.max_resized_height)
        resized_width = int(resized_height * aspect_ratio)
        if width != resized_width:
            logger.debug(f"resized ({width}, {height}) to: ({resized_width}, {resized_height})")
            input_image = input_image.resize((resized_width, resized_height))

    input_image = np.asarray(input_image)
    input_image = input_image.astype(np.float32)

    input_image = input_image[:, :, [2, 1, 0]]

    if expand_dim:
        input_image = np.expand_dims(input_image, axis=0)
    return input_image

# convert from flatten to the final image matrix
def post_processing(transformed_image):
    if not type(transformed_image) == np.ndarray:
        transformed_image = transformed_image.numpy()
    transformed_image = transformed_image[0]
    transformed_image = transformed_image[:, :, [2, 1, 0]]
    transformed_image = transformed_image * 0.5 + 0.5
    transformed_image = transformed_image * 255
    return transformed_image


def save_transformed_image(output_image, img_filename, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    transformed_image_path = os.path.join(save_dir, img_filename)

    if output_image is not None:
        image = PIL.Image.fromarray(output_image.astype("uint8"))
        image.save(transformed_image_path)

    return transformed_image_path

# to display the cartoon and real image in a comparison view (grid, horizontal, vertical)
def save_concatenated_image(image_paths, image_folder="comparison", num_columns=2):
    images = [PIL.Image.open(i).convert('RGB') for i in image_paths]
    # pick the image which is the smallest, and resize the others to match it (can be arbitrary image shape here)
    min_shape = sorted([(np.sum(i.size), i.size) for i in images])[0][1]
    array = np.asarray([np.asarray(i.resize(min_shape)) for i in images])

    view = args.comparison_view
    if view == "smart":
        width, height = min_shape[0], min_shape[1]
        aspect_ratio = width / height
        logger.debug(f"(width, height): ({width}, {height}), aspect_ratio: {aspect_ratio}")
        grid_suitable = (len(args.styles) + 1) % num_columns == 0
        is_portrait = aspect_ratio <= 0.75
        if grid_suitable and not is_portrait:
            view = "grid"
        elif is_portrait:
            view = "horizontal"
        else:
            view = "vertical"

    if view == "horizontal":
        images_comb = np.hstack(array)
    elif view == "vertical":
        images_comb = np.vstack(array)
    elif view == "grid":
        rows = np.split(array, num_columns)
        rows = [np.hstack(row) for row in rows]
        images_comb = np.vstack([row for row in rows])
    else:
        logger.debug(f"Wrong `comparison_view`: {args.comparison_view}")

    images_comb = PIL.Image.fromarray(images_comb)
    file_name = image_paths[0].split("/")[-1]

    if args.output_dir not in image_folder:
        image_folder = os.path.join(args.output_dir, image_folder)
    if not os.path.exists(image_folder):
        os.makedirs(image_folder)

    image_path = os.path.join(image_folder, file_name)
    images_comb.save(image_path)
    return image_path


def transform_png_images(image_paths, model, style, return_existing_result=False):
    transformed_image_paths = list()
    save_dir = os.path.join("/".join(image_paths[0].split("/")[:-1]), style)
    logger.debug(f"Transforming {len(image_paths)} images and saving them to {save_dir}....")

    if return_existing_result:
        return glob.glob(os.path.join(save_dir, "*.png"))

    num_batch = int(np.ceil(len(image_paths) / args.batch_size))
    image_paths = np.array_split(image_paths, num_batch)

    logger.debug(f"Processing {num_batch} batches with batch_size={args.batch_size}...")
    for batch_image_paths in image_paths:
        image_filenames = [path.split("/")[-1] for path in batch_image_paths]
        input_images = [pre_processing(path, expand_dim=False) for path in batch_image_paths]
        input_images = np.stack(input_images, axis=0)
        transformed_images = model(input_images)
        output_images = [post_processing(image, style=style)
                         for image in np.split(transformed_images, transformed_images.shape[0])]
        paths = [save_transformed_image(img, f, save_dir)
                 for img, f in zip(output_images, image_filenames)]
        transformed_image_paths.extend(paths)

    return transformed_image_paths


def result_exist(image_path, style):
    return os.path.exists(os.path.join(args.output_dir, style, image_path.split("/")[-1]))


def cli_cartoonization():

    start = datetime.now()
    logger.info(f"Transformed images will be saved to `{args.output_dir}` folder.")
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # create temporary folder which will be deleted after transformations
    if not os.path.exists(TEMPORARY_DIR):
        os.makedirs(TEMPORARY_DIR)

    # decide what styles to used in this execution
    styles = STYLES if args.all_styles else args.styles

    models = list()
    for style in styles:
        models.append(cartoongan.load_model(style))

    logger.info(f"Cartoonizing images using {', '.join(styles)} style...")

    image_paths = []
    for ext in VALID_EXTENSIONS:
        image_paths.extend(glob.glob(os.path.join(args.input_dir, f"*.{ext}")))
    logger.info(f"Preparing to transform {len(image_paths)} images from `{args.input_dir}` directory...")

    progress_bar = tqdm(image_paths, desc='Transforming')
    for image_path in progress_bar:
        image_filename = image_path.split("/")[-1]
        progress_bar.set_postfix(File=image_filename)
        related_image_paths = [image_path]
        for model, style in zip(models, styles):
            input_image = pre_processing(image_path)
            save_dir = os.path.join(args.output_dir, style)
            return_existing_result = result_exist(image_path, style) and not args.overwrite

            if not return_existing_result:
                transformed_image = model(input_image)
                output_image = post_processing(transformed_image, style=style)
                transformed_image_path = save_transformed_image(output_image, image_filename, save_dir)
            else:
                transformed_image_path = save_transformed_image(None, image_filename, save_dir)

            related_image_paths.append(transformed_image_path)

        if not args.skip_comparison:
            save_concatenated_image(related_image_paths)
    progress_bar.close()

    time_elapsed = datetime.now() - start
    logger.info(f"Total processing time: {time_elapsed}")
def gui_cartoonization(style, filepath, filename):
    # the interaction with the client may be in the form of socket programming
    # in any case, the raw input image must be saved in client_image variable
    # and the style name will be saved in a style variable
    start = datetime.now()

    model = cartoongan.load_model(style)

    input_image = pre_processing(filepath)
    transformed_image = model(input_image)
    output_image = post_processing(transformed_image)
    save_dir = \
        os.path.join(
        "C:\\Users\\psham\\PycharmProjects\\Cartoonization\\output_images", style, "\\input_images")
    transformed_image_path = save_transformed_image(output_image, filename, save_dir)
    time_elapsed = datetime.now() - start
    return transformed_image_path, time_elapsed
# if __name__ == "__main__":
#     cli_cartoonization()
