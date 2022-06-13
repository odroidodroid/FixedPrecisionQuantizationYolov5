import json
import os


if __name__ =='__main__' :

    # gt_dir = '/home/youngjin/datasets/coco/annotations/instances_val2017.json'
    # with open(gt_dir) as fgt :
    #     gt_json_dict = json.load(fgt)
    #     print(len(gt_json_dict))
    #     print(gt_json_dict.keys())
    #     print(gt_json_dict['annotations'])
    
    result_json_dict = []
    result_dir = '/home/youngjin/projects/runs/detect/exp4/labels'
    result_list = os.listdir(result_dir)
    result_json = str(f'/home/youngjin/projects/runs/detect/exp4/result.json')

    for res in result_list :
        image_id = int(res.split('.')[0])
        f = open(result_dir +'/'+ res,'r')
        lines = f.readlines()
        for line in lines :
            value = line.split(' ')
            bbox_value = []
            for v in range(1,4) :
                bbox_value.append(float(value[v]))
            result_json_dict.append({
                'image_id' : image_id,
                'category_id' : int(value[0]),
                'bbox' : bbox_value,
                'score' : 1.0
            })
        f.close()

    with open(result_json,'w') as f:
        json.dump(result_json_dict, f)