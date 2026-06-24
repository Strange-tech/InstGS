
import os
import sys
from torchvision.utils import save_image

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
from plyfile import PlyData, PlyElement
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
from argparse import ArgumentParser, Namespace

from arguments import ModelParams, PipelineParams, OptimizationParams
from scene import Scene, GaussianModel
from gaussian_renderer import render

from dds_cloudapi_sdk import Config
from dds_cloudapi_sdk import Client
from dds_cloudapi_sdk.tasks.v2_task import V2Task, create_task_with_local_image_auto_resize


from sam2.build_sam import build_sam2_video_predictor, build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from utils_sam2.video_utils import create_video_from_images
from utils_sam2.common_utils import CommonUtils
from utils_sam2.mask_dictionary_model import MaskDictionaryModel, ObjectInfo

from copy import deepcopy
from tqdm import tqdm


SCENE_NAME = "tomato"

MODEL_PATH = f"./output/{SCENE_NAME}"

FEATURE_GAUSSIAN_ITERATION = 10000
# Set up command line argument parser
parser = ArgumentParser(description="Training script parameters")
lp = ModelParams(parser)
op = OptimizationParams(parser)
pp = PipelineParams(parser)
parser.add_argument("--ip", type=str, default="127.0.0.1")
parser.add_argument("--port", type=int, default=np.random.randint(10000, 20000))
parser.add_argument("--debug_from", type=int, default=-1)
parser.add_argument("--detect_anomaly", action="store_true", default=False)
parser.add_argument(
    "--test_iterations", nargs="+", type=int, default=[7_000, 30_000]
)
parser.add_argument(
    "--save_iterations", nargs="+", type=int, default=[7_000, 10_000]
)
parser.add_argument("--quiet", action="store_true")
parser.add_argument(
    "--checkpoint_iterations", nargs="+", type=int, default=[10_000, 20_000, 30_000]
)
parser.add_argument("--start_checkpoint", type=str, default=None)
args = parser.parse_args(sys.argv[1:])
args.save_iterations.append(args.iterations)

args.source_path = f"./data/{SCENE_NAME}"
args.model_path = f"./output/{SCENE_NAME}"
args.iterations = 10000

dataset = lp.extract(args)
opt = op.extract(args)
pipe = pp.extract(args)

scene_gaussians = GaussianModel(dataset.sh_degree)
scene_gaussians.load_ply(f"{MODEL_PATH}/point_cloud/iteration_30000/point_cloud.ply")

scene = Scene(
    dataset,
    scene_gaussians,
)
cameras = scene.getTrainCameras()
print("There are", len(cameras), "views in the dataset.")

bg_color = [1, 1, 1]
background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

# FOR DEBUGGING: 查看3dgs渲染图像
img = render(cameras[0], scene_gaussians, pipe, background)["render"]
save_image(img, f"./data/{SCENE_NAME}/original.png")

# FOR DEBUGGING: 查看GT图像
img = cameras[23].original_image * 255
img = img.permute([1, 2, 0]).detach().cpu().numpy().astype(np.uint8)
Image.fromarray(img).save(f"./tmp/{23}.jpg")


############################# SAM2 #####################################


"""
Step 1: Environment settings and model initialization
"""
# use bfloat16 for the entire notebook
torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

if torch.cuda.get_device_properties(0).major >= 8:
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# init sam image predictor and video predictor model
sam2_checkpoint = "./checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
device = "cuda" if torch.cuda.is_available() else "cpu"
print("device", device)

video_predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint)
sam2_image_model = build_sam2(model_cfg, sam2_checkpoint, device=device)
image_predictor = SAM2ImagePredictor(sam2_image_model)

# init grounding dino model from huggingface
model_id = "IDEA-Research/grounding-dino-tiny"
processor = AutoProcessor.from_pretrained(model_id)
grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(
    device
)

# setup the input image and text prompt for SAM 2 and Grounding DINO
# VERY important: text queries need to be lowercased + end with a dot

# text = "chair. table. vase."
# text = (
#     "candlestick. cylindrical jar. black bowl. tall bottles with pipes. small bottle."
# )
text = "tomato"
BOX_THRESHOLD = 0.2
IOU_THRESHOLD = 0.8
GROUNDING_MODEL = "GroundingDino-1.6-Pro"  # 使用字符串替代枚举值

# `video_dir` a directory of JPEG frames with filenames like `<frame_index>.jpg`
# video_dir = "/root/autodl-tmp/data/houses/images/"
video_dir = f"./data/{SCENE_NAME}/images/"
# 'output_dir' is the directory to save the annotated frames
# output_dir = "./outputs"
output_dir = f"./data/{SCENE_NAME}/sam_masks/"
# 'output_video_path' is the path to save the final video
output_video_path = f"{output_dir}/output.mp4"
# create the output directory
CommonUtils.creat_dirs(output_dir)
mask_data_dir = os.path.join(output_dir, "mask_data")
json_data_dir = os.path.join(output_dir, "json_data")
result_dir = os.path.join(output_dir, "result")
CommonUtils.creat_dirs(output_dir)
CommonUtils.creat_dirs(mask_data_dir)
CommonUtils.creat_dirs(json_data_dir)
# scan all the JPEG frame names in this directory
frame_names = [
    p
    for p in os.listdir(video_dir)
    if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG", ".png", ".PNG"]
]
# frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))
frame_names.sort(key=lambda p: int(os.path.splitext(p)[0].replace("frame", "")))

# init video predictor state
inference_state = video_predictor.init_state(video_path=video_dir)
step = 100  # the step to sample frames for Grounding DINO predictor

sam2_masks = MaskDictionaryModel()
PROMPT_TYPE_FOR_VIDEO = "mask"  # box, mask or point
objects_count = 0

"""
Step 2: Prompt Grounding DINO and SAM image predictor to get the box and mask for all frames
"""
print("Total frames:", len(frame_names))
begin_idx = 0
end_idx = len(frame_names)
for start_frame_idx in range(begin_idx, end_idx, step):
    # prompt grounding dino to get the box coordinates on specific frame
    print("start_frame_idx", start_frame_idx)
    # continue
    img_path = os.path.join(video_dir, frame_names[start_frame_idx])
    image = Image.open(img_path)
    image_base_name = frame_names[start_frame_idx].split(".")[0]
    mask_dict = MaskDictionaryModel(
        promote_type=PROMPT_TYPE_FOR_VIDEO, mask_name=f"mask_{image_base_name}.npy"
    )

    # run Grounding DINO 1.5 on the image

    API_TOKEN_FOR_GD1_5 = "a99b0b26caf732610c735945a75de422"

    config = Config(API_TOKEN_FOR_GD1_5)
    # Step 2: initialize the client
    client = Client(config)

    # Create task with local image auto resized (Recommended: faster processing)
    task = create_task_with_local_image_auto_resize(
        api_path="/v2/task/grounding_dino/detection",
        api_body_without_image={
            "model": GROUNDING_MODEL,
            "prompt": {"type": "text", "text": text},
            "targets": ["bbox"],
            "bbox_threshold": BOX_THRESHOLD,
            "iou_threshold": IOU_THRESHOLD,
        },
        image_path=img_path,
    )

    # image_url = client.upload_file(img_path)
    # task = V2Task(
    #     api_path="/v2/task/grounding_dino/detection",
    #     api_body={
    #         "model": GROUNDING_MODEL,
    #         "image": image_url,
    #         "prompt": {"type": "text", "text": text},
    #         "targets": ["bbox"],
    #         "bbox_threshold": BOX_THRESHOLD,
    #         "iou_threshold": IOU_THRESHOLD,
    #     },
    # )

    client.run_task(task)
    result = task.result

    objects = result["objects"]  # the list of detected objects
    input_boxes = []
    confidences = []
    class_names = []

    for idx, obj in enumerate(objects):
        input_boxes.append(obj["bbox"])
        confidences.append(obj["score"])
        class_names.append(obj["category"])

    input_boxes = np.array(input_boxes)
    OBJECTS = class_names
    if input_boxes.shape[0] != 0:
        # prompt SAM image predictor to get the mask for the object
        image_predictor.set_image(np.array(image.convert("RGB")))

        # prompt SAM 2 image predictor to get the mask for the object
        masks, scores, logits = image_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_boxes,
            multimask_output=False,
        )
        # convert the mask shape to (n, H, W)
        if masks.ndim == 2:
            masks = masks[None]
            scores = scores[None]
            logits = logits[None]
        elif masks.ndim == 4:
            masks = masks.squeeze(1)

        """
        Step 3: Register each object's positive points to video predictor
        """

        # If you are using point prompts, we uniformly sample positive points based on the mask
        if mask_dict.promote_type == "mask":
            mask_dict.add_new_frame_annotation(
                mask_list=torch.tensor(masks).to(device),
                box_list=torch.tensor(input_boxes),
                label_list=OBJECTS,
            )
        else:
            raise NotImplementedError("SAM 2 video predictor only support mask prompts")

        objects_count = mask_dict.update_masks(
            tracking_annotation_dict=sam2_masks,
            iou_threshold=IOU_THRESHOLD,
            objects_count=objects_count,
        )
        print("objects_count", objects_count)

    else:
        print(
            "No object detected in the frame, skip merge the frame merge {}".format(
                frame_names[start_frame_idx]
            )
        )
        mask_dict = sam2_masks

    """
        Step 4: Propagate the video predictor to get the segmentation results for each frame
    """
    if len(mask_dict.labels) == 0:
        mask_dict.save_empty_mask_and_json(
            mask_data_dir,
            json_data_dir,
            image_name_list=frame_names[start_frame_idx : start_frame_idx + step],
        )
        print(
            "No object detected in the frame, skip the frame {}".format(start_frame_idx)
        )
        continue
    else:
        video_predictor.reset_state(inference_state)

        for object_id, object_info in mask_dict.labels.items():
            frame_idx, out_obj_ids, out_mask_logits = video_predictor.add_new_mask(
                inference_state,
                start_frame_idx,
                object_id,
                object_info.mask,
            )

        video_segments = {}  # output the following {step} frames tracking masks
        for (
            out_frame_idx,
            out_obj_ids,
            out_mask_logits,
        ) in video_predictor.propagate_in_video(
            inference_state,
            max_frame_num_to_track=step,
            start_frame_idx=start_frame_idx,
        ):
            frame_masks = MaskDictionaryModel()

            for i, out_obj_id in enumerate(out_obj_ids):
                out_mask = out_mask_logits[i] > 0.0  # .cpu().numpy()
                object_info = ObjectInfo(
                    instance_id=out_obj_id,
                    mask=out_mask[0],
                    class_name=mask_dict.get_target_class_name(out_obj_id),
                )
                object_info.update_box()
                frame_masks.labels[out_obj_id] = object_info
                image_base_name = frame_names[out_frame_idx].split(".")[0]
                frame_masks.mask_name = f"mask_{image_base_name}.npy"
                frame_masks.mask_height = out_mask.shape[-2]
                frame_masks.mask_width = out_mask.shape[-1]

            video_segments[out_frame_idx] = frame_masks
            # sam2_masks = deepcopy(frame_masks)

        print("video_segments:", len(video_segments))

labels = video_segments[0].labels
class_names = [l.class_name for l in labels.values()]
print("class_names", class_names)

######################### Gradient Computer #############################

import glob
import torch.nn.functional as F


def load_all_masks(folder):
    pt_files = glob.glob(os.path.join(folder, "*.pt"))
    data_list = []
    for f in pt_files:
        try:
            data = torch.load(f, map_location="cuda")  # 也可以是 "cuda"
            data_list.append(data)
        except Exception as e:
            print(f"❌ Failed to load {f}: {e}")
    return data_list


def get_masked_gradients(gaussians, frame, mask_2d):
    """
    gaussians: 3D Gaussians with learnable parameters (e.g., xyz, shs, opacity)
    frame: output from render(), containing 'image' and 'viewspace_points'
    mask_2d: (H, W) 2D bool mask indicating target region
    """

    # 1. 获取渲染图像和 viewspace points（像素 ←→ 高斯点索引）
    rendered_image = frame["render"]
    viewspace_points = frame[
        "viewspace_points"
    ]  # shape: [N, 2] (x, y) for each Gaussian
    visibility_filter = frame["visibility_filter"]  # bool mask of visible Gaussians

    # 2. 创建 mask 图像，只对掩码区域内像素计算 loss
    # print(mask_2d.shape)
    mask_2d = mask_2d.float().unsqueeze(0).unsqueeze(0)  # [N,C,H,W]
    mask_resized = F.interpolate(
        mask_2d, size=(rendered_image.shape[1], rendered_image.shape[2]), mode="nearest"
    ).squeeze(0)
    mask_resized = mask_resized.to(rendered_image.device)
    mask_3c = mask_resized.expand_as(rendered_image)  # [3, H, W]

    # 3. 构造 masked loss
    loss = (rendered_image * mask_3c).sum()

    # 4. 启用梯度追踪
    viewspace_points.retain_grad()
    loss.backward(retain_graph=True)

    # 5. 收集有梯度的高斯点（贡献了像素）
    if viewspace_points.grad is None:
        raise RuntimeError("viewspace_points.grad is None")

    per_point_grad = viewspace_points.grad.norm(dim=-1)  # shape: [N]
    visible_grad = torch.zeros_like(per_point_grad)
    visible_grad[visibility_filter] = per_point_grad[visibility_filter]

    return visible_grad


inst_grads = {}

mask_path = f"./data/{SCENE_NAME}/instance_mask"
os.makedirs(mask_path, exist_ok=True)

for frame_idx, frame_masks_info in tqdm(
    video_segments.items(), total=len(video_segments), desc="Gradient frames"
):
    mask = frame_masks_info.labels
    all_masks = []
    for obj_id, obj_info in mask.items():
        if obj_id not in inst_grads:
            inst_grads[obj_id] = torch.zeros(
                scene_gaussians.get_xyz.shape[0], device="cuda"
            )

        view = deepcopy(cameras[frame_idx])
        frame = render(view, scene_gaussians, pipe, background)

        mask_2d = obj_info.mask
        all_masks.append(mask_2d)

        inst_grads[obj_id] += get_masked_gradients(scene_gaussians, frame, mask_2d)
        inst_grads[obj_id] -= 0.1 * get_masked_gradients(
            scene_gaussians, frame, ~mask_2d
        )
    all_masks = torch.stack(all_masks, dim=0)
    all_masks = torch.any(all_masks.bool(), dim=0)
    # print(all_masks.shape)
    torch.save(all_masks, f'{mask_path}/{frame_names[frame_idx].split(".")[0]}.pt')

print(inst_grads)

# pre_color = torch.ones((scene_gaussians.get_xyz.shape[0], 3), device="cuda")
all_grads = torch.zeros(scene_gaussians.get_xyz.shape[0], device="cuda")

for obj_id, grad in inst_grads.items():
    all_grads += (grad > 0).float()
    # if class_names[obj_id - 1] == "table":
    #     pre_color[grad > 0] = torch.tensor(
    #         [31 / 255, 119 / 255, 180 / 255], device="cuda"
    #     )
    # elif class_names[obj_id - 1] == "chair":
    #     pre_color[grad > 0] = torch.tensor(
    #         [255 / 255, 127 / 255, 14 / 255], device="cuda"
    #     )
    # elif class_names[obj_id - 1] == "vase":
    #     pre_color[grad > 0] = torch.tensor(
    #         [44 / 255, 160 / 255, 44 / 255], device="cuda"
    #     )

    # s_g = deepcopy(scene_gaussians)
    # s_g.segment(grad > 0)
    # s_g.save_ply(f"/root/autodl-tmp/data/{SCENE_NAME}/seg_inst/{obj_id}.ply")

# s_g = deepcopy(scene_gaussians)
# s_g.segment(all_grads > 0)
# s_g.save_ply(f"/root/autodl-tmp/data/{SCENE_NAME}/seg_inst/instances.ply")

# rendered_img = render(
#     cameras[83],
#     s_g,
#     pipeline.extract(args),
#     background,
#     # filtered_mask=all_grads > 0,
#     # override_color=pre_color,
# )["render"]
# save_image(rendered_img, f"/root/autodl-tmp/data/{SCENE_NAME}/segmented_view.png")

s_g = deepcopy(scene_gaussians)
s_g.segment(all_grads <= 0)

rendered_img = render(cameras[83], s_g, pipe, background)["render"]
save_image(rendered_img, f"./data/{SCENE_NAME}/bg.png")
s_g.save_ply(f"./data/{SCENE_NAME}/seg_inst/bg.ply")
