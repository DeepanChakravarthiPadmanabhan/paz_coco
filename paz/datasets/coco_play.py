from paz.datasets import COCODataset

COCO_DATASET_PATH = '/media/deepan/externaldrive1/datasets_project_repos/mscoco'

data_splits = ['train', 'test']
data_names = ['train2017', 'val2017']

data = COCODataset(COCO_DATASET_PATH,
                   split='train',
                   class_names='all',
                   name='train2017')
data_loaded = data.load_data()
print(len(data_loaded))

# loading datasets
data_managers, datasets, evaluation_data_managers = [], [], []
for data_name, data_split in zip(data_names, data_splits):
    data_manager = COCODataset(COCO_DATASET_PATH, data_split, name=data_name)
    data_managers.append(data_manager)
    datasets.append(data_manager.load_data())
    if data_split == 'test':
        eval_data_manager = COCODataset(
            COCO_DATASET_PATH, data_split, name=data_name, evaluate=True)
        evaluation_data_managers.append(eval_data_manager)
