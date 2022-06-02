import json
import os


if __name__ =='__main__' :
    result_json_dict = []
    result_dir = './PTQ/runs/detect/exp/labels'
    result_list = os.listdir(result_dir)
    result_json = str(f'./PTQ/runs/detect/exp/result.json')

    for res in result_list :
        image_id = int(res.split('.')[0])
        f = open(result_dir +'/'+ res,'r')
        lines = f.readlines()
        for line in lines :
            value = line.split(' ')
            bbox_value = []
            for v in range(1,4) :
                bbox_value.append(value[v])
            result_json_dict.append({
                'image_id' : image_id,
                'category_id' : value[0],
                'bbox' : bbox_value,
                'score' : 1.0
            })
        f.close()

    with open(result_json,'w') as f:
        json.dump(result_json_dict, f)