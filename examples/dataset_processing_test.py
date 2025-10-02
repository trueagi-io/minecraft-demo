from mcdemoaux.logging.binary_saver import *
from mcdemoaux.logging.binary_loader import *

# saver = ImageHDF5Saver('dataset/')
# saver.save_to_hdf5('output_images.h5')
#
# loader = ImageHDF5Loader('output_images.h5')
#
# # Get info
# print("Dataset shape:", loader.get_shape())
#
# # Load all
# images = loader.get_all()
#
# # Load a batch
# batch = loader.get_batch(0, 10)
#
# # close the file
# loader.close()

saver = ImageWithMetadataHDF5Saver('dataset/')
saver.save_to_hdf5('dataset_with_metadata.h5')

loader = ImageWithMetadataHDF5Loader('dataset_with_metadata.h5')

# Load all
images, metadata = loader.get_all()
print("Loaded", len(images), "images")

# Load some batch
batch_images, batch_meta = loader.get_batch(0, 10)
print(batch_meta)

# Load a single item
image, meta = loader.get_item(0)
print(meta)

# Close the file when done
loader.close()
