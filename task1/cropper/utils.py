import json


def get_full_boxes(gt_json: str, pred_path: str = 'full_boxes.json') -> None:
    with open(gt_json, 'r') as f:
        images = json.load(f)['images']

    json_content = []
    for image in images:
        image_data = {
            'bbox': [0, 0, image['width'], image['height']],
            'category_id': 1,
            'image_id': image['id'],
            'score': 1.0
        }
        json_content.append(image_data)

    with open(pred_path, 'w', encoding='utf-8') as json_file:
        json.dump(json_content, json_file, indent=None)


if __name__ == '__main__':
    get_full_boxes(r'..\data\train_gt.json', 'train_boxes.json')
    get_full_boxes(r'..\data\test_gt.json', 'test_boxes.json')
