from skimage import io, transform
from torch.utils.data import Dataset
import pandas as pd
import os
from PIL import Image, ImageFilter

class DogBreeds(Dataset):
    '''
    The dog breed dataset from Kaggle.
    https://www.kaggle.com/c/dog-breed-identification
    '''

    def __init__(self, csv_file, root_dir, transform=None, data_split='train'):
        """
        Args:
            pandas dataframe (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.data_split = data_split
        self.unique_labels = None
        self.dogs_dataframe = self.breeds_to_int(csv_file)
        
    def __len__(self):
        return len(self.dogs_dataframe)

    def __getitem__(self, idx):
        if self.data_split == 'submission':
            img_name = os.path.join(self.root_dir,'test', self.dogs_dataframe.ix[idx, 0]+'.jpg')
        else:
            img_name = os.path.join(self.root_dir,'train', self.dogs_dataframe.ix[idx, 0]+'.jpg')
        image = Image.fromarray(io.imread(img_name))
        label = self.dogs_dataframe.ix[idx, 1]

        if self.transform:
            image = self.transform(image)

        return {'image': image, 'label': label}
    
    def breeds_to_int(self, csv_file):
        
        # CSV file is of the form: imagename, dog breed
        labels_frame = pd.read_csv(os.path.join(self.root_dir, csv_file))
        
        if not self.data_split == 'submission':
            
            ## Need to convert dog breed names to integers
            self.unique_labels = sorted(list(set(labels_frame.ix[:,1]))) # get unique breeds

            # We create a reference dictionary 
            label_to_int_dict = {k: v for v, k in enumerate(self.unique_labels)}

            # Change labels to int
            labels_as_int = [label_to_int_dict[x] for x in labels_frame.ix[:,1]]

            # Reassemble the data frame with the integers values as the labels
            del labels_frame['breed']
            labels_frame['breed'] = labels_as_int

            # Split the data
            end_train_idx = int(0.8 * labels_frame.shape[0])
            end_valid_idx = int(0.9 * labels_frame.shape[0])
        
        if self.data_split == 'train':
            data_frame = labels_frame.iloc[:end_train_idx, :].reset_index(drop=True)
        elif self.data_split == 'valid':
            data_frame = labels_frame.iloc[end_train_idx: end_valid_idx, :].reset_index(drop=True)
        elif self.data_split == 'test':
            data_frame = labels_frame.iloc[end_valid_idx: , :].reset_index(drop=True)
        elif self.data_split == 'submission':
            data_frame = labels_frame  
        else:
            raise ValueError("Data split should be one of: ['train', 'valid', 'split', 'submission']")
        
        # Return the correct label, and the two transforming dictionaries
        return data_frame