##
import os

def del_images(image_path, label_path) :
    image_list = os.listdir(image_path)
    label_list = os.listdir(label_path)

    del_image_list = []

    for index, image in enumerate(image_list) : 
        try :
            if label_list[index] :
                pass
        except :
            del_image_list.append(image)
    del_list_len = len(del_image_list)
    for index in range(del_list_len) :
        os.remove(image_path + '/' + del_image_list[index])       

def main() :
    del_images('/home/youngjin/datasets/coco/val/images', '/home/youngjin/datasets/coco/val/labels')


if __name__ == "__main__" :
    main()