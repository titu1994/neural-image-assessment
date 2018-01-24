
from image_preprocessing import ImageDataGenerator, randomCropFlips, \
    centerCrop
generator = ImageDataGenerator(preprocessing_function=randomCropFlips())
batch_size = 64


def get_batches(x, y, gen=generator, shuffle=True, batch_size=batch_size,):
    return gen.flow_from_filelist(x, y,
                          shuffle=shuffle, batch_size=batch_size, image_size=image_size,
                                  cropped_image_size=cropped_image_size)
    # save_to_dir=r's:/temp/genimg', save_format='jpg')


generator_valid = ImageDataGenerator(preprocessing_function=centerCrop())


def get_batches_valid(x, y, gen=generator_valid, shuffle=False,
                      batch_size=batch_size,
                      image_size=PRE_CROP_IMAGE_SIZE, cropped_image_size=IMAGE_SIZE):
    return gen.flow_from_filelist(x, y,
                          shuffle=shuffle, batch_size=batch_size, image_size=image_size,
                                  cropped_image_size=cropped_image_size)
    # save_to_dir=r's:/temp/genimg', save_format='jpg')

train_generator = get_batches(dataset.train_image_paths, dataset.train_scores,
                              batch_size=batch_size)
validation_generator = get_batches_valid(dataset.test_image_paths, dataset.test_scores,
                                         batch_size=batch_size)