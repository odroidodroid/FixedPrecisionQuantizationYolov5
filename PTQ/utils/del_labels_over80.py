import os

DEL_category_id = ['12', '26' , '29', '30', '45', '66', '68', '69', '71', '83', '91']

def del_labels(label_path) :
    label_list = os.listdir(label_path)

    del_label_list = []

    for _, label in enumerate(label_list) :
        with open(label_path+'/'+label, 'r') as fp :
            lines = fp.readlines()
            count = 0
            for line in lines :
                values = line.split(' ')
                if values[0] in DEL_category_id :
                    del_label_list.append((label, count))
                count +=1

    del_list_len = len(del_label_list)

    for index in range(del_list_len) :
        with open(label_path + '/' + del_label_list[index][0], 'r') as fo :
           lines = fo.readlines()
           newlines = []
           for i, line in enumerate(lines) :
               if i != del_label_list[index][1] :
                   newlines.append(line)
        
        os.remove(label_path + '/' + del_label_list[index][0])
    
        with open(label_path + '/' + del_label_list[index][0], 'w') as fo :
            fo.writelines(newlines)





def main() :
    del_labels('/home/youngjin/datasets/coco/val/labels')

if __name__ == '__main__' :
    main()