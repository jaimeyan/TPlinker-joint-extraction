import os, json, requests

current_path = '/'.join(os.path.dirname(__file__).split('/')[:-1])


def lis_item_word_parse(examinationItem):
    if 'char_list' not in examinationItem or ('lis_item_name' not in examinationItem and 'name' not in examinationItem) or (
            'char_probs' not in examinationItem and 'char_prob_list' not in examinationItem):
        return None
    char_list = examinationItem['char_list']
    char_prob_list = examinationItem['char_probs'] if 'char_probs' in examinationItem else examinationItem[
        'char_prob_list']
    name = examinationItem['lis_item_name'] if 'lis_item_name' in examinationItem else examinationItem['name']
    start_pos = ''.join(char_list).find(name)
    end_pos = start_pos + len(name)
    char_list = char_list[start_pos:end_pos]
    char_prob_list = char_prob_list[start_pos:end_pos]
    return char_list, char_prob_list, name


def fetch_lis_items_from_pic(image_path, data_json):
    print('processing pic:', image_path.split(r'/')[-1])
    ocr_url = 'http://192.168.99.130:51000/ocr/inspection_rectify_extract'
    files = {"img": open(os.path.join(current_path, image_path), 'rb'),
             'data': open(os.path.join(current_path, data_json), 'rb')}
    headers = {}
    response = requests.request("POST", ocr_url, headers=headers, files=files)
    if response.status_code != 200:
        print("request ocr error!")
    json_data = json.loads(response.content)
    if json_data['code'] == 200:
        if json_data['data']['examinationList']:
            res = []
            for i in json_data['data']['examinationList']:
                parsed_res = lis_item_word_parse(i)
                if not parsed_res:
                    continue
                res.append({'char_list': parsed_res[0], 'char_prob_list': parsed_res[1], 'name': parsed_res[2]})
            return res
    return None


if __name__ == '__main__':
    print(fetch_lis_items_from_pic('../pics/huayandan_pics/img_936.jpeg', data_json='ocr_request.json'))
