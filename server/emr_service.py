from flask import Flask, request
from predict.predictor import Predictor
import uuid
import torch
import json
import config
import cv2, os, requests, json
from numpy import mean
from collections import defaultdict
from tqdm import tqdm


# image = cv2.imread('../data/pics/化验单-1.jpg')
# print(image.shape)
# maxx = image.shape[1]
# maxy = image.shape[0]
# print(maxx)
# draw1 = cv2.rectangle(image, (int(0.2 * maxx), int(0.2 * maxy)), (int(0.8 * maxx), int(0.8 * maxy)),
#                       (0, 0, 255), 3)
# cv2.imwrite('./test_pics/test_化验单-1.jpg', draw1)
# # cv2.imshow('test', draw1)

def cal_inter_area(miny1, maxy1, miny2, maxy2):
    maxy = max(maxy1, maxy2)
    miny = min(miny1, miny2)
    inter_h = max(min(maxy1, maxy2) - max(miny1, miny2), 0)
    h1, h2 = maxy1 - miny1 + 1, maxy2 - miny2 + 1
    inter_ratio = inter_h / min(h1, h2)
    return inter_ratio


def get_rects_from_xys(item_list, threshold=0.8):
    row_set = {}
    for item in item_list:
        need_new_line = True
        x1, x2 = item['text_coord'][0], item['text_coord'][2]
        y1, y2 = item['text_coord'][1], item['text_coord'][3]
        if not row_set:
            row_set[(x1, x2, y1, y2)] = {'items': [item], 'text': ''}
            continue
        for key in row_set:
            inter_ratio = cal_inter_area(key[-2], key[-1], y1, y2)
            if inter_ratio >= threshold:
                row_set[key]['items'].append(item)
                need_new_line = False
                new_key = (min([i['text_coord'][0] for i in row_set[key]['items']]),
                           max([i['text_coord'][2] for i in row_set[key]['items']]),
                           min([i['text_coord'][1] for i in row_set[key]['items']]),
                           max([i['text_coord'][3] for i in row_set[key]['items']]))
                row_set[new_key] = row_set.pop(key)
                break
        if need_new_line:
            row_set[(x1, x2, y1, y2)] = {'items': [item], 'text': ''}

    key_sort = sorted(row_set.keys(), key=lambda x: x[2])
    res_row_set = defaultdict()
    for row in key_sort:
        res_row_set[row] = defaultdict()
        res_row_set[row]['items'] = sorted(row_set[row]['items'], key=lambda x: x['text_coord'][0])
        res_row_set[row]['text'] = ' '.join([i['text_string'] for i in res_row_set[row]['items']])

    return res_row_set


def fetch_ocr_items_from_pic(image_path, data_json):
    print('processing pic:', image_path.split(r'/')[-1])
    ocr_url = 'http://192.168.99.130:51000/ai/ocr/inspection_extract'
    files = {"file": open(os.path.join(current_path, image_path), 'rb'),
             'data': open(os.path.join(current_path, data_json), 'rb')}
    headers = {}
    response = requests.request("POST", ocr_url, headers=headers, files=files)
    if response.status_code != 200:
        print("request ocr error!")
    json_data = json.loads(response.content)
    if json_data['data']['img_data_list'][0]['extract_info']['texts']:
        texts = json_data['data']['img_data_list'][0]['extract_info']['texts']
    else:
        return None
    return texts


def sample2output(sample):
    attr_dict = {'检验项目-结果', '检验项目-参考范围', '检验项目-单位', '检验项目-代号'}
    basic_dict = {'姓名', '年龄', '床号', '性别', '科室', '收样时间', '报告时间', '检验时间'}
    basicinfo, lis_list = [], []
    lis_index = {}
    idx = 0
    for i in sample['entity_list']:
        if i['type'] == '检验项目':
            lis_index[i['text']] = idx
            idx += 1
            lis_list.append(
                {
                    "abbr": "",
                    "brX": 0,
                    "brY": 0,
                    "char_list": list(i['text']),
                    "char_prob_list": ["0"] * len(i['text']),
                    "interestAbbr": None,
                    "interestName": None,
                    "ltX": 0,
                    "ltY": 0,
                    "matchConfidence": None,
                    "matchQueryName": None,
                    "name": i['text'],
                    "refer": "",
                    "referMax": None,
                    "referMin": None,
                    "standardAbbr": None,
                    "standardName": None,
                    "standardUnit": None,
                    "unit": "",
                    "value": "",
                    "value2": None,
                    "value_refer": 0
                }
            )
        elif i['type'] in basic_dict:
            basicinfo.append(
                {
                    "content": i['text'],
                    "key": i['type'],
                    "maxX": 0,
                    "maxY": 0,
                    "minX": 0,
                    "minY": 0
                }
            )

    for i in sample['relation_list']:
        if i['predicate'] == '收样时间标签-收样时间':
            basicinfo.append(
                {
                    "content": i['subject'],
                    "key": '收样时间',
                    "maxX": 0,
                    "maxY": 0,
                    "minX": 0,
                    "minY": 0
                }
            )
            continue
        elif i['predicate'] == '报告时间标签-报告时间':
            basicinfo.append(
                {
                    "content": i['subject'],
                    "key": '报告时间',
                    "maxX": 0,
                    "maxY": 0,
                    "minX": 0,
                    "minY": 0
                }
            )
            continue
        elif i['predicate'] == '检验时间标签-检验时间':
            basicinfo.append(
                {
                    "content": i['subject'],
                    "key": '检验时间',
                    "maxX": 0,
                    "maxY": 0,
                    "minX": 0,
                    "minY": 0
                }
            )
            continue
        elif i['predicate'] in attr_dict:
            lis_num = lis_index[i['object']]
            tar_lis = lis_list[lis_num]
            if i['predicate'] == '检验项目-参考范围':
                tar_lis['refer'] = i['subject']
            elif i['predicate'] == '检验项目-结果':
                tar_lis['value'] = i['subject']
            elif i['predicate'] == '检验项目-单位':
                tar_lis['unit'] = i['subject']
            elif i['predicate'] == '检验项目-代号':
                tar_lis['abbr'] = i['subject']

    return basicinfo, lis_list


app = Flask(__name__)

config = config.eval_config
MyPredictor = Predictor(config)
model_state_path = config["model_state_path"]
MyPredictor.rel_extractor.load_state_dict(torch.load(model_state_path))
MyPredictor.rel_extractor.eval()


@app.route('/ai/nlp/tp/ocr_emr/huayandan_emr/', methods=['POST'])
def emr_server():
    ocr_response = request.get_data()
    ocr_response = json.loads(ocr_response)

    text_info = ocr_response['data']['img_data_list'][0]['text_info']
    row_set = get_rects_from_xys(text_info, 0.8)
    text = ' '.join([i['text'] for _, i in row_set.items()])
    pic_name = ocr_response['data']['ori_file_name']
    failed_pages = ocr_response['data']['failed_pages']
    success_pages = ocr_response['data']['success_pages']
    errors = ocr_response['errors']
    detect_img_name = ocr_response['data']['img_data_list'][0]['detect_img_name']
    detect_img_path = ocr_response['data']['img_data_list'][0]['detect_img_path']

    ori_data = [{
        "id": pic_name,
        "text": text,
        "relation_list": [],
        "entity_list": []
    }]

    pred_sample_list, df_al = MyPredictor.predict(ori_data)
    basicinfo, lis_list = sample2output(pred_sample_list[0])
    extract_info = {'basicInfos': basicinfo, 'examinationList': lis_list}
    request_id = "req-" + str(uuid.uuid4())
    bag = {
        "code": 200,
        "data":
            {
                "failed_pages": failed_pages,
                "img_data_list": [
                    {
                        "detect_img_name": detect_img_name,
                        "detect_img_path": detect_img_path,
                        "extract_infos": extract_info,
                        "height": 0,
                        "rotate_angle": 0.0,
                        "status_code": 200,
                        "width": 0
                    }
                ],
                "msg": "success",
                "ori_file_name": pic_name,
                "request_id": request_id,
                "success_pages": success_pages
            },
        "errors": errors,
        "success": True
    }
    res = json.dumps(bag, ensure_ascii=False)
    return res


app.run(host='0.0.0.0', port=1000, debug=True)
