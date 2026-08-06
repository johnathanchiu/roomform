# Samples

## Pre-baked result (no setup needed)

`redwood-apartment/` is a complete pipeline output — boundary
prediction, SpatialLM objects, RGB display cloud — committed so the
debug viewer works out of the box:

    uv run python viewer/serve.py --artifacts samples

## Input scans

Source scans are not committed (hundreds of MB). The pipeline runs on
any registered indoor point cloud ply; two good public-domain starters
from the [Redwood Indoor LiDAR-RGBD dataset](http://redwood-data.org/indoor_lidar_rgbd/)
(merged & resampled laser point clouds):

    uvx gdown 0B-ePgl6HF260a3MtSWJIY1pBSG8   # apartment (0.5 GB)
    uvx gdown 0B-ePgl6HF260N0tCdzA2Zm5JdTQ   # bedroom (0.2 GB)

then

    uv run python -m roomform.pipe.e2e <downloaded>.ply

Attribution: Park, Zhou & Koltun, *Colored Point Cloud Registration
Revisited*, ICCV 2017 — dataset released into the public domain.
