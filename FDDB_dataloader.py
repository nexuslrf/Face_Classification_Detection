import numpy as np
import PIL.Image as Image
import PIL.ImageDraw as ImageDraw
import cv2
import torch.utils.data

class FDDB(torch.utils.data.Dataset):
    def __init__(self, datalist, transform = None, crop=True, zero_one=False):
        super(type(self), self).__init__()
        self.datalist = np.load(datalist).tolist()
        self.directions = np.array([
           [ 1,  0,  1,  0],
           [ 0,  1,  0,  1],
           [-1,  0, -1,  0],
           [ 0, -1,  0, -1],
           [ 1,  1,  1,  1],
           [-1, -1, -1, -1],
           [ 1, -1,  1, -1],
           [-1,  1, -1,  1],
           [ 0,  0,  0,  0]
        ])
        self.transform = transform
        self.crop = crop
        self.zero_one = zero_one
    
    def __len__(self):
        return len(self.datalist)
    
    def __getitem__(self, idx):
        img, pos = self.datalist[idx]
        pos = int(pos)
        img = Image.open(img)
        if self.crop:
            w,h = img.size
            wi,hi = w/5, h/5
            offset = np.array([wi,hi,wi,hi])
            bndbox = np.array([wi,hi,4*wi,4*hi])
            bndbox = bndbox + self.directions[pos] * offset
            img = img.crop(bndbox)
        if self.transform is not None:
            img = self.transform(img)
        label = 1 if pos == 8 else (0 if self.zero_one else -1)
        return img, label
        
    def draw_bndbox(self, idx, mypos=None): # input PIL image
        img, pos = self.datalist[idx]
        pos = int(pos) if mypos is None else mypos
        img = Image.open(img)
        w,h = img.size
        wi,hi = w/5, h/5
        offset = np.array([wi,hi,wi,hi])
        bndbox = np.array([wi,hi,4*wi,4*hi])
        bndbox = bndbox + self.directions[pos] * offset * 0.99
        bndbox = bndbox.astype(int)
        # Draw!
        draw = ImageDraw.Draw(img)
        draw.line(bndbox[[0,1,2,1,2,3,0,3,0,1]].tolist(), fill='red')
        return img
    
def main():
    db = FDDB('train_list.npy',crop=False)
    print(len(db))

if __name__=='__main__':
    main()