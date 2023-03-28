import json
import os

from mmcv import Config
from mmpose.apis import init_pose_model, inference_top_down_pose_model
from tqdm.auto import tqdm


def infer(config_path: str, data_dir: str, gt_json: str, pred_path: str = 'predictions.json',
          device: str = 'cuda') -> None:
    cfg = Config.fromfile(config_path)
    model = init_pose_model(cfg, fr'workdirs\hrnet_1\latest.pth', device=device)

    with open(os.path.join(data_dir, gt_json), 'r') as f:
        images = json.load(f)['images']

    json_content = {}
    for image in tqdm(images):
        image_results, _ = inference_top_down_pose_model(model, os.path.join(data_dir, image['file_name']))
        if len(image_results) > 0:
            best_result = image_results[0]  # max(image_results, key=lambda x: x['bbox'][4])
            points = best_result['keypoints'].astype(float)
            points[:, 0] = points[:, 0] / image['width']
            points[:, 1] = points[:, 1] / image['height']
            key_format = image['file_name'].split('\\', 1)[1]
            key_format = key_format.replace('png', 'json').replace('images', 'ground_truth').replace('\\', '|')
            json_content[key_format] = points[:, :2].tolist()

    pred_dir = os.path.dirname(pred_path)
    os.makedirs(pred_dir, exist_ok=True)
    with open(pred_path, 'w', encoding='utf-8') as json_file:
        json.dump(json_content, json_file, indent=None)
