from path import Path

# get the AVA dataset from here:
# http://academictorrents.com/details/71631f83b11d3d79d8f84efe0a7e12f0ac001460
model_prefix = "nasnet"

base_images_path = Path('AVA_dataset/images').abspath()
if not base_images_path.exists():
    base_images_path = Path(r's:\datasets\AVA_dataset\images')
base_dir = Path(__file__).dirname()
dataset_path = base_dir.joinpath('dataset')
weights_dir = base_dir.joinpath('weights')
weights_file = weights_dir.joinpath(model_prefix + '_weights.h5')
weights_epoch_file = weights_dir.joinpath(
    model_prefix + '_weights_{epoch:02d}-{val_loss:.3f}.h5')
weights_top_file = weights_dir.joinpath(model_prefix+'_top_weights.h5')
log_dir = base_dir.joinpath('logs')
temp_dir = base_dir.joinpath('temp')
if not temp_dir.exists():
    temp_dir.makedirs_p()
bc_path_features = temp_dir.joinpath('features_{}.bc'.format(model_prefix))
scores_path = temp_dir.joinpath('scores_{}.bc'.format(model_prefix))

PRE_CROP_IMAGE_SIZE = 256
IMAGE_SIZE = 224
target_size = (IMAGE_SIZE, IMAGE_SIZE)

assert base_images_path.exists()
assert dataset_path.exists()
