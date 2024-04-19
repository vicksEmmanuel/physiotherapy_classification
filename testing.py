from util.config import CFG
from util.load_dataloader import prepare_ava_dataset
from pytorchvideo.data.clip_sampling import ClipInfo, RandomClipSampler
from pytorchvideo.data.ava import TimeStampClipSampler;


prepare_ava_dataset('train', CFG)


x = RandomClipSampler(5)

print(x._clip_duration)

(
    clip_start,
    clip_end,
    clip_index,
    aug_index,
    is_last_clip,
) = x.__call__(None,384.6666666666667,  None)

print(float(clip_start), float(clip_end), clip_index, aug_index, is_last_clip, ' ===')


# y = TimeStampClipSampler(x)

# t = y.__call__(None,384.6666666666667, None)

# print(t)
