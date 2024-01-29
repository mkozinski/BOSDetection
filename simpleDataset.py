from torch.utils.data import Dataset

class SimpleDataset(Dataset):

  def __init__(self, img, lbl, augment=lambda i,l:(i,l),
    get_img=lambda i:i, get_lbl=lambda l:l):

    # img and lbl are sequences
    # they can store the data or file names

    self.img=img
    self.lbl=lbl
    self.augment=augment
    self.get_img=get_img
    self.get_lbl=get_lbl

  def __len__(self):
    return min(len(self.lbl),len(self.img))

  def __getitem__(self, idx):

    img=self.get_img(self.img[idx])
    lbl=self.get_lbl(self.lbl[idx])

    i,l=self.augment(img,lbl)
    return  i,l

